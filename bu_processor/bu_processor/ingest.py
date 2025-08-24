#!/usr/bin/env python3
"""
🚀 BACKGROUND PDF INGESTION SYSTEM
=================================

Production-ready background job processing system for PDF ingestion with:
- Async job processing with retry logic
- Classification metadata storage
- Comprehensive error handling and logging
- Job status tracking and monitoring
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import tempfile
import traceback

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .integrations.pinecone_facade import PineconeManager

# DRY RUN Support für Testing
DRY_RUN = os.getenv("DRY_RUN_INGEST", "false").lower() == "true"

# Import BU-Processor components
from .core.config import get_config
from .pipeline.classifier import RealMLClassifier, extract_text_from_pdf
from .embeddings.embedder import Embedder
from .integrations.pinecone_facade import make_pinecone_manager
from .storage.sqlite_store import SqliteStore

logger = structlog.get_logger("ingest")

# Hilfsfunktionen für robuste Klassifikations-Ergebnis-Verarbeitung
def _as_dict(obj):
    """Normalisiert Klassifikations-Ergebnisse zu einem Dict"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return getattr(obj, "__dict__", {}) or {}

def _extract_classification_fields(classification_result: dict):
    """Robuste Extraktion aller Klassifikationsfelder mit fallback auf verschiedene Feldnamen"""
    # Kategorie/Label - verschiedene mögliche Feldnamen berücksichtigen
    category = (classification_result.get("category") or 
               classification_result.get("predicted_category") or
               classification_result.get("label") or
               classification_result.get("category_label") or
               classification_result.get("predicted_label"))
    
    # Expliziter Label-Wert (falls unterschiedlich von category)
    label = (classification_result.get("predicted_label") or
             classification_result.get("label") or
             classification_result.get("category_label") or
             category)  # fallback auf category
    
    # Confidence - verschiedene mögliche Feldnamen
    confidence = (classification_result.get("confidence") or
                 classification_result.get("predicted_confidence") or
                 classification_result.get("score") or
                 0.0)
    
    # All scores - für detaillierte Analyse
    all_scores = (classification_result.get("all_scores") or
                 classification_result.get("scores") or
                 classification_result.get("probabilities") or
                 {})
    
    # Page count - falls vorhanden
    page_count = (classification_result.get("page_count") or
                 classification_result.get("pages") or
                 classification_result.get("num_pages") or
                 None)
    
    return {
        "predicted_label": label,
        "predicted_category": category,
        "confidence": float(confidence) if confidence is not None else 0.0,
        "all_scores": all_scores,
        "page_count": page_count
    }

def _assert_index_dimension(pinecone_manager, embed_dim: int):
    """Preflight-Check: Prüfe ob Embedding-Dimension zur Index-Dimension passt"""
    try:
        index_dim = pinecone_manager.get_index_dimension()
    except Exception as e:
        logger.warning("Could not check index dimension", error=str(e))
        # Nicht kritisch - lass den Upsert versuchen
        return
    
    if index_dim is not None and index_dim != embed_dim:
        raise RuntimeError(
            f"Dimension mismatch: Pinecone index expects {index_dim} dimensions, "
            f"but embedding model produces {embed_dim} dimensions. "
            f"This would cause upsert failures."
        )
    
    if index_dim is not None:
        logger.debug("Dimension check passed", 
                    index_dim=index_dim, 
                    embedding_dim=embed_dim)

class ClassificationError(RuntimeError):
    """Exception für Klassifikations-Fehler"""
    pass

class JobStatus(str, Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class IngestJob(BaseModel):
    """Background ingestion job model"""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str
    filename: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    result: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class JobManager:
    """Manages background ingestion jobs"""
    
    def __init__(self):
        self.config = get_config()
        self.jobs: Dict[str, IngestJob] = {}
        self.classifier: Optional[RealMLClassifier] = None
        self.pinecone_manager: Optional["PineconeManager"] = None
        self.embedder: Optional[Embedder] = None
        self.storage: Optional[SqliteStore] = None
        
    async def _initialize_components(self):
        """Lazy initialization of heavy components"""
        if self.classifier is None:
            logger.info("Initializing classifier for background jobs")
            self.classifier = RealMLClassifier()
            
        if self.embedder is None:
            logger.info("Initializing embedder for background jobs")
            self.embedder = Embedder()
            
        if self.pinecone_manager is None and self.config.vector_db.enable_vector_db:
            logger.info("Initializing Pinecone manager for background jobs")
            self.pinecone_manager = make_pinecone_manager(
                index_name=self.config.vector_db.pinecone_index_name,
                api_key=self.config.vector_db.pinecone_api_key,
                environment=self.config.vector_db.pinecone_env,   # v2
                cloud=self.config.vector_db.pinecone_cloud,       # v3
                region=self.config.vector_db.pinecone_region,     # v3
                namespace=self.config.vector_db.pinecone_namespace
            )
            # Ensure index exists with correct dimension
            if self.embedder:
                self.pinecone_manager.ensure_index(dimension=self.embedder.dimension)
            
        if self.storage is None:
            logger.info("Initializing storage for background jobs")
            self.storage = SqliteStore()
    
    def create_job(self, file_path: str, filename: str) -> IngestJob:
        """Create a new ingestion job"""
        job = IngestJob(
            file_path=file_path,
            filename=filename
        )
        self.jobs[job.job_id] = job
        logger.info("Created ingestion job", 
                   job_id=job.job_id, 
                   filename=filename)
        return job
    
    def get_job(self, job_id: str) -> Optional[IngestJob]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[IngestJob]:
        """List all jobs"""
        return list(self.jobs.values())
    
    async def process_job(self, job_id: str) -> IngestJob:
        """Process a single ingestion job"""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Update job status
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        
        try:
            logger.info("Starting job processing", 
                       job_id=job_id, 
                       filename=job.filename)
            
            # Initialize components
            await self._initialize_components()
            
            # Process the PDF
            result = await self._process_pdf(job)
            
            # Update job with success
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.result = result
            
            logger.info("Job completed successfully", 
                       job_id=job_id,
                       processing_time=(job.completed_at - job.started_at).total_seconds())
            
        except ClassificationError as e:
            # Spezifische Behandlung für Klassifikations-Fehler
            logger.error("Classification failed - job will be retried", 
                        job_id=job_id, 
                        filename=job.filename,
                        error=str(e))
            
            job.error_message = f"Classification failed: {str(e)}"
            job.retry_count += 1
            
            # Determine if we should retry
            if job.retry_count <= job.max_retries:
                job.status = JobStatus.RETRYING
                logger.info("Classification error - job will be retried", 
                           job_id=job_id, 
                           retry_count=job.retry_count,
                           max_retries=job.max_retries,
                           stage="classification")
                
                # Schedule retry with exponential backoff
                retry_delay = 2 ** job.retry_count  # 2, 4, 8 seconds
                await asyncio.sleep(retry_delay)
                return await self.process_job(job_id)
            else:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                logger.error("Classification failed after max retries", 
                            job_id=job_id,
                            retry_count=job.retry_count,
                            stage="classification")
        
        except FileNotFoundError as e:
            # Datei-Fehler sind meist nicht retry-fähig
            logger.error("File not found - job failed permanently", 
                        job_id=job_id, 
                        filename=job.filename,
                        error=str(e))
            
            job.error_message = f"File not found: {str(e)}"
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            # Kein Retry für File-Not-Found
        
        except ValueError as e:
            # ValueError (z.B. kein Text im PDF) - retry kann helfen
            logger.error("Value error in job processing - will retry", 
                        job_id=job_id, 
                        filename=job.filename,
                        error=str(e))
            
            job.error_message = f"Processing error: {str(e)}"
            job.retry_count += 1
            
            if job.retry_count <= job.max_retries:
                job.status = JobStatus.RETRYING
                logger.info("Value error - job will be retried", 
                           job_id=job_id, 
                           retry_count=job.retry_count,
                           max_retries=job.max_retries,
                           stage="processing")
                
                retry_delay = 2 ** job.retry_count
                await asyncio.sleep(retry_delay)
                return await self.process_job(job_id)
            else:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                logger.error("Processing failed after max retries", 
                            job_id=job_id,
                            retry_count=job.retry_count,
                            stage="processing")
        
        except Exception as e:
            # Allgemeine Exceptions - mit vollem Traceback loggen
            logger.exception("Unexpected job processing error - will retry", 
                           job_id=job_id, 
                           filename=job.filename,
                           error=str(e))
            
            job.error_message = f"Unexpected error: {str(e)}"
            job.retry_count += 1
            
            # Determine if we should retry
            if job.retry_count <= job.max_retries:
                job.status = JobStatus.RETRYING
                logger.info("Unexpected error - job will be retried", 
                           job_id=job_id, 
                           retry_count=job.retry_count,
                           max_retries=job.max_retries,
                           stage="general")
                
                # Schedule retry with exponential backoff
                retry_delay = 2 ** job.retry_count  # 2, 4, 8 seconds
                await asyncio.sleep(retry_delay)
                return await self.process_job(job_id)
            else:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                logger.error("Job failed after max retries", 
                            job_id=job_id,
                            retry_count=job.retry_count,
                            stage="general")
        
        return job
    
    async def _process_pdf(self, job: IngestJob) -> Dict[str, Any]:
        """Process a single PDF file with comprehensive stage logging"""
        file_path = Path(job.file_path)
        job_id = job.job_id
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # STAGE: Text Extraction
        logger.info("Stage start", job_id=job_id, stage="extract", file=str(file_path))
        text_content = extract_text_from_pdf(str(file_path))
        
        if not text_content or len(text_content.strip()) < 10:
            raise ValueError("PDF contains no extractable text or text too short")
        
        logger.info("Stage done", job_id=job_id, stage="extract", 
                   text_length=len(text_content), 
                   char_count=len(text_content.strip()))
        
        # STAGE: Classification
        logger.info("Stage start", job_id=job_id, stage="classification")
        raw_classification_result = self.classifier.classify_text(text_content)
        
        # HARTER CONTRACT: Klassifikations-Ergebnis strikt prüfen (JobManager._process_pdf)
        classification_result = _as_dict(raw_classification_result)
        
        # Fehler-Check: Wenn error gesetzt ist -> Exception
        if classification_result.get("error"):
            raise ClassificationError(f"Classification failed: {classification_result['error']}")
        
        # ROBUSTE Feldextraktion mit fallback auf verschiedene Feldnamen (JobManager)
        extracted_fields = _extract_classification_fields(classification_result)
        
        if not extracted_fields["predicted_label"] and not extracted_fields["predicted_category"]:
            raise ClassificationError("Classification returned no label/category.")
        
        # Robuste Metadaten-Befüllung mit geprüften Werten
        predicted_label = extracted_fields["predicted_label"] or extracted_fields["predicted_category"]
        predicted_category = extracted_fields["predicted_category"] or extracted_fields["predicted_label"]
        predicted_confidence = extracted_fields["confidence"]
        
        logger.info("Stage done", job_id=job_id, stage="classification", 
                   label=predicted_label, 
                   category=predicted_category,
                   confidence=predicted_confidence)
        
        # Prepare metadata with consistent structure
        metadata = {
            "filename": job.filename,
            "file_path": str(file_path),
            "processed_at": datetime.now().isoformat(),
            "job_id": job.job_id,
            "processing_type": "background_job",
            
            # Core classification metadata for filtering/analysis - GEPRÜFTE WERTE
            "predicted_label": predicted_label,
            "predicted_category": predicted_category,
            "predicted_confidence": predicted_confidence,
            
            # Detailed classification results
            "classification": {
                "predicted_label": predicted_label,
                "predicted_category": predicted_category,
                "confidence": predicted_confidence,
                "all_scores": extracted_fields["all_scores"],
                "text_length": len(text_content),
                "model_labels": self.classifier.get_available_labels() or [],
                "model_info": {
                    "model_dir": getattr(self.classifier, 'model_dir', None),
                    "labels_source": "artifact_labels.txt" if self.classifier.get_available_labels() else "fallback"
                }
            }
        }
        
        # STAGE: Database Storage
        if self.storage:
            logger.info("Stage start", job_id=job_id, stage="database")
            doc_id = await self._store_document(text_content, metadata)
            metadata["document_id"] = doc_id
            logger.info("Stage done", job_id=job_id, stage="database", document_id=doc_id)
        
        # STAGE: Pinecone Storage
        if self.pinecone_manager and self.config.pinecone.enabled:
            logger.info("Stage start", job_id=job_id, stage="embed_upsert")
            await self._store_in_pinecone(text_content, metadata)
            logger.info("Stage done", job_id=job_id, stage="embed_upsert")
        
        return {
            "status": "success",
            "metadata": metadata,
            "text_length": len(text_content),
            "classification": classification_result
        }
    
    async def _store_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """Store document in SQLite database"""
        # Robuste Feldzugriffe für doc_data
        classification_data = metadata.get("classification", {})
        
        # Create document entry
        doc_data = {
            "content": content,
            "metadata": json.dumps(metadata),
            "source": metadata.get("filename", "unknown"),
            "document_type": classification_data.get("predicted_label", "unknown")
        }
        
        # Store document
        doc_id = self.storage.add_document(
            content=content,
            metadata=metadata,
            source=metadata.get("filename", "unknown")
        )
        
        # Update with classification metadata - ROBUSTE Feldzugriffe
        classification_data = metadata.get("classification", {})
        classification_metadata = {
            "classification_confidence": classification_data.get("confidence", 0.0),
            "classification_scores": classification_data.get("all_scores", {}),
            "processing_job_id": metadata.get("job_id", "unknown")
        }
        
        self.storage.update_document_metadata(doc_id, classification_metadata)
        
        return doc_id
    
    async def _store_in_pinecone(self, content: str, metadata: Dict[str, Any]):
        """Store document embeddings in Pinecone using the new simplified API"""
        try:
            if not self.pinecone_manager or not self.embedder:
                logger.warning("Pinecone manager or embedder not initialized, skipping vector storage")
                return
            
            # Simple chunking for now (could be enhanced)
            chunk_size = 1000
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            
            if not chunks:
                logger.warning("No chunks to process for Pinecone storage")
                return
            
            # Generate embeddings for all chunks using the new embedder
            logger.info("Generating embeddings for chunks", count=len(chunks))
            embeddings = self.embedder.encode(chunks)
            
            # Prepare data for upsert_vectors (new API signature)
            ids = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_text": chunk[:200] + "..." if len(chunk) > 200 else chunk
                }
                
                vector_id = f"{metadata.get('job_id', 'unknown')}_chunk_{i}"
                ids.append(vector_id)
                metadatas.append(chunk_metadata)
            
            # Use new upsert_vectors API with separate arrays
            job_id = metadata.get('job_id', 'unknown')
            
            # DRY RUN Check
            if DRY_RUN:
                logger.info("DRY_RUN mode - skipping Pinecone upsert", 
                           job_id=job_id, 
                           chunks=len(chunks),
                           embed_dim=len(embeddings[0]) if embeddings else 0)
                return
            
            logger.info("Upserting vectors to Pinecone", job_id=job_id, count=len(ids))
            
            # Use the new simplified API
            result = self.pinecone_manager.upsert_vectors(
                ids=ids,
                vectors=embeddings,
                metadatas=metadatas
            )
            
            logger.info("Successfully stored document in Pinecone", 
                       job_id=job_id,
                       chunks_stored=len(ids),
                       result=result)
            
        except Exception as e:
            job_id = metadata.get('job_id', 'unknown')
            logger.exception("Failed to store document in Pinecone", 
                           job_id=job_id,
                           stage="embed_upsert",
                           error=str(e))
            raise

# Global job manager instance
_job_manager = None

def get_job_manager() -> JobManager:
    """Get or create global job manager instance"""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager

# Background task functions for FastAPI
async def process_pdf_background(file_path: str, filename: str) -> str:
    """Background task for PDF processing"""
    job_manager = get_job_manager()
    job = job_manager.create_job(file_path, filename)
    
    # Process job asynchronously
    await job_manager.process_job(job.job_id)
    
    return job.job_id

# Unified PDF processing function for both API and CLI
def process_pdf(file_path: str, output_dir: Optional[str] = None, 
                store_in_pinecone: bool = True, store_in_sqlite: bool = True) -> Dict[str, Any]:
    """
    Universal PDF processing function for both API and CLI usage.
    
    Args:
        file_path: Path to the PDF file
        output_dir: Optional output directory for extracted content
        store_in_pinecone: Whether to store embeddings in Pinecone 
        store_in_sqlite: Whether to store document in SQLite
        
    Returns:
        Dict containing processing results, classification, and storage info
    """
    from pathlib import Path
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    logger.info("Starting unified PDF processing", 
               filename=file_path.name,
               store_pinecone=store_in_pinecone,
               store_sqlite=store_in_sqlite)
    
    try:
        # 1. Extract text from PDF
        logger.info("Extracting text from PDF", filename=file_path.name)
        text_content = extract_text_from_pdf(str(file_path))
        
        if not text_content or len(text_content.strip()) < 10:
            raise ValueError("PDF contains no extractable text or text too short")
        
        # 2. Initialize classifier
        config = get_config()
        classifier = RealMLClassifier(config)
        
        # 3. Classify the document
        logger.info("Classifying document", filename=file_path.name)
        raw_classification_result = classifier.classify_text(text_content)
        
        # HARTER CONTRACT: Klassifikations-Ergebnis strikt prüfen
        classification_result = _as_dict(raw_classification_result)
        
        # Fehler-Check: Wenn error gesetzt ist -> Exception
        if classification_result.get("error"):
            raise ClassificationError(f"Classification failed: {classification_result['error']}")
        
        # ROBUSTE Feldextraktion mit fallback auf verschiedene Feldnamen (process_pdf)
        extracted_fields = _extract_classification_fields(classification_result)
        
        if not extracted_fields["predicted_label"] and not extracted_fields["predicted_category"]:
            raise ClassificationError("Classification returned no label/category.")
        
        # Robuste Metadaten-Befüllung mit geprüften Werten
        predicted_label = extracted_fields["predicted_label"] or extracted_fields["predicted_category"]
        predicted_category = extracted_fields["predicted_category"] or extracted_fields["predicted_label"]
        predicted_confidence = extracted_fields["confidence"]
        
        # 4. Prepare enhanced metadata with deterministic labels
        metadata = {
            "filename": file_path.name,
            "file_path": str(file_path.absolute()),
            "processed_at": datetime.now().isoformat(),
            "processing_type": "unified_process_pdf",
            
            # Core classification metadata for filtering/analysis - GEPRÜFTE WERTE
            "predicted_label": predicted_label,
            "predicted_category": predicted_category,
            "predicted_confidence": predicted_confidence,
            
            # Detailed classification results
            "classification": {
                "predicted_label": predicted_label,
                "predicted_category": predicted_category,
                "confidence": predicted_confidence,
                "all_scores": extracted_fields["all_scores"],
                "text_length": len(text_content),
                "model_labels": classifier.get_available_labels() or [],
                "model_info": {
                    "model_dir": getattr(classifier, 'model_dir', None),
                    "labels_source": "artifact_labels.txt" if classifier.get_available_labels() else "fallback"
                }
            },
            "storage": {
                "pinecone_enabled": store_in_pinecone,
                "sqlite_enabled": store_in_sqlite
            }
        }
        
        # 5. Store results
        storage_results = {}
        
        # SQLite storage
        if store_in_sqlite:
            try:
                logger.info("Storing document in SQLite", filename=file_path.name)
                storage = SqliteStore(config)
                doc_id = storage.add_document(
                    content=text_content,
                    metadata=metadata,
                    source=file_path.name
                )
                storage_results["sqlite"] = {
                    "status": "success",
                    "document_id": doc_id
                }
                metadata["document_id"] = doc_id
                logger.info("SQLite storage completed", document_id=doc_id)
            except Exception as e:
                logger.error("SQLite storage failed", error=str(e))
                storage_results["sqlite"] = {
                    "status": "error", 
                    "error": str(e)
                }
        
        # Pinecone storage  
        if store_in_pinecone and config.pinecone.enabled:
            try:
                logger.info("Storing embeddings in Pinecone", filename=file_path.name)
                pinecone_manager = make_pinecone_manager(
                    index_name=config.pinecone.index_name,
                    api_key=config.pinecone.api_key,
                    environment=config.pinecone.environment,   # v2
                    cloud=config.pinecone.cloud,               # v3
                    region=config.pinecone.region,             # v3
                    namespace=config.pinecone.namespace
                )
                
                # Simple chunking strategy
                chunk_size = 1000
                chunks = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]
                
                if chunks:
                    # Generate embeddings for all chunks at once
                    logger.info("Generating embeddings for chunks", count=len(chunks))
                    embeddings = asyncio.run(pinecone_manager.generate_embeddings_async(chunks, show_progress=False))
                    
                    # PREFLIGHT CHECK: Prüfe Dimension-Kompatibilität vor Upsert
                    if embeddings and len(embeddings) > 0:
                        embed_dim = len(embeddings[0])
                        _assert_index_dimension(pinecone_manager, embed_dim)
                    
                    # Prepare vectors for batch upsert
                    vectors = []
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        chunk_metadata = {
                            **metadata,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "chunk_text": chunk[:200] + "..." if len(chunk) > 200 else chunk
                        }
                        
                        # Create unique ID for chunk
                        chunk_id = f"{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_chunk_{i}"
                        vectors.append((chunk_id, embedding, chunk_metadata))
                    
                    # Batch upsert all vectors - robuste Signatur-Behandlung
                    if DRY_RUN:
                        logger.info("DRY_RUN mode - skipping Pinecone upsert", 
                                   filename=file_path.name,
                                   chunks=len(vectors),
                                   embed_dim=len(embeddings[0]) if embeddings else 0)
                        vectors_stored = len(vectors)  # Simulate successful upsert
                    else:
                        logger.info("Upserting vectors to Pinecone", count=len(vectors))
                        try:
                            # Versuch 1: Standard Tupel-Format (id, vector, metadata)
                            result = pinecone_manager.upsert_vectors(vectors)
                        except TypeError:
                            # Versuch 2: Falls items-Format erwartet wird
                            logger.debug("Retrying with items format")
                            items = []
                            for chunk_id, embedding, chunk_metadata in vectors:
                                items.append({
                                    "id": chunk_id,
                                    "values": embedding,
                                    "metadata": chunk_metadata
                                })
                            result = pinecone_manager.upsert_vectors(vectors=items)
                        
                        vectors_stored = result.get("upserted", 0)
                else:
                    vectors_stored = 0
                
                storage_results["pinecone"] = {
                    "status": "success",
                    "chunks_stored": vectors_stored,
                    "total_chunks": len(chunks)
                }
                logger.info("Pinecone storage completed", vectors_stored=vectors_stored)
                
            except Exception as e:
                logger.exception("Pinecone storage failed", 
                               filename=file_path.name,
                               stage="embed_upsert",
                               error=str(e))
                storage_results["pinecone"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # 6. Save extracted content to file (if output_dir specified)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save text content
            text_file = output_path / f"{file_path.stem}_extracted.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # Save metadata as JSON
            metadata_file = output_path / f"{file_path.stem}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    **metadata,
                    "storage_results": storage_results
                }, f, indent=2, ensure_ascii=False)
            
            storage_results["files"] = {
                "text_file": str(text_file),
                "metadata_file": str(metadata_file)
            }
        
        # 7. Final result - robuste Feldzugriffe
        processed_at_str = metadata.get("processed_at")
        if processed_at_str:
            try:
                processing_time = (datetime.now() - datetime.fromisoformat(processed_at_str.replace('Z', '+00:00'))).total_seconds()
            except (ValueError, AttributeError):
                processing_time = 0.0
        else:
            processing_time = 0.0
            
        result = {
            "status": "success",
            "filename": file_path.name,
            "text_length": len(text_content),
            "classification": classification_result,
            "metadata": metadata,
            "storage": storage_results,
            "processing_time": processing_time
        }
        
        logger.info("Unified PDF processing completed", 
                   filename=file_path.name,
                   classification=predicted_label,
                   confidence=predicted_confidence)
        
        return result
        
    except Exception as e:
        logger.exception("Unified PDF processing failed", 
                        filename=file_path.name, 
                        error=str(e))
        raise

def cleanup_old_jobs(max_age_hours: int = 24):
    """Cleanup old completed/failed jobs"""
    job_manager = get_job_manager()
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    jobs_to_remove = []
    for job_id, job in job_manager.jobs.items():
        if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED] and 
            job.completed_at and job.completed_at < cutoff_time):
            jobs_to_remove.append(job_id)
    
    for job_id in jobs_to_remove:
        del job_manager.jobs[job_id]
        logger.info("Cleaned up old job", job_id=job_id)
    
    logger.info("Job cleanup completed", removed_jobs=len(jobs_to_remove))
