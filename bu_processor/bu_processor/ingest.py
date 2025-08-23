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
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import traceback

import structlog
from pydantic import BaseModel, Field

# Import BU-Processor components
from .core.config import get_config
from .pipeline.classifier import RealMLClassifier, extract_text_from_pdf
from .pipeline.pinecone_integration import PineconeManager
from .storage.sqlite_store import SqliteStore

logger = structlog.get_logger("ingest")

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
        self.pinecone_manager: Optional[PineconeManager] = None
        self.storage: Optional[SqliteStore] = None
        
    async def _initialize_components(self):
        """Lazy initialization of heavy components"""
        if self.classifier is None:
            logger.info("Initializing classifier for background jobs")
            self.classifier = RealMLClassifier()
            
        if self.pinecone_manager is None and self.config.pinecone.enabled:
            logger.info("Initializing Pinecone manager for background jobs")
            self.pinecone_manager = PineconeManager()
            
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
            
        except Exception as e:
            logger.error("Job processing failed", 
                        job_id=job_id, 
                        error=str(e),
                        traceback=traceback.format_exc())
            
            job.error_message = str(e)
            job.retry_count += 1
            
            # Determine if we should retry
            if job.retry_count <= job.max_retries:
                job.status = JobStatus.RETRYING
                logger.info("Job will be retried", 
                           job_id=job_id, 
                           retry_count=job.retry_count,
                           max_retries=job.max_retries)
                
                # Schedule retry with exponential backoff
                retry_delay = 2 ** job.retry_count  # 2, 4, 8 seconds
                await asyncio.sleep(retry_delay)
                return await self.process_job(job_id)
            else:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                logger.error("Job failed after max retries", 
                            job_id=job_id,
                            retry_count=job.retry_count)
        
        return job
    
    async def _process_pdf(self, job: IngestJob) -> Dict[str, Any]:
        """Process a single PDF file"""
        file_path = Path(job.file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text from PDF
        logger.info("Extracting text from PDF", filename=job.filename)
        text_content = extract_text_from_pdf(str(file_path))
        
        if not text_content or len(text_content.strip()) < 10:
            raise ValueError("PDF contains no extractable text or text too short")
        
        # Classify the document
        logger.info("Classifying document", filename=job.filename)
        classification_result = self.classifier.classify_text(text_content)
        
        # Prepare metadata with consistent structure
        metadata = {
            "filename": job.filename,
            "file_path": str(file_path),
            "processed_at": datetime.now().isoformat(),
            "job_id": job.job_id,
            "processing_type": "background_job",
            
            # Core classification metadata for filtering/analysis
            "predicted_label": classification_result["predicted_label"],
            "predicted_category": classification_result.get("predicted_category", classification_result["predicted_label"]),
            "predicted_confidence": classification_result["confidence"],
            
            # Detailed classification results
            "classification": {
                "predicted_label": classification_result["predicted_label"],
                "predicted_category": classification_result.get("predicted_category", classification_result["predicted_label"]),
                "confidence": classification_result["confidence"],
                "all_scores": classification_result.get("all_scores", {}),
                "text_length": len(text_content),
                "model_labels": self.classifier.get_available_labels() or [],
                "model_info": {
                    "model_dir": getattr(self.classifier, 'model_dir', None),
                    "labels_source": "artifact_labels.txt" if self.classifier.get_available_labels() else "fallback"
                }
            }
        }
        
        # Store in database
        if self.storage:
            logger.info("Storing document in database", filename=job.filename)
            doc_id = await self._store_document(text_content, metadata)
            metadata["document_id"] = doc_id
        
        # Store in Pinecone if enabled
        if self.pinecone_manager and self.config.pinecone.enabled:
            logger.info("Storing embeddings in Pinecone", filename=job.filename)
            await self._store_in_pinecone(text_content, metadata)
        
        return {
            "status": "success",
            "metadata": metadata,
            "text_length": len(text_content),
            "classification": classification_result
        }
    
    async def _store_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """Store document in SQLite database"""
        # Create document entry
        doc_data = {
            "content": content,
            "metadata": json.dumps(metadata),
            "source": metadata["filename"],
            "document_type": metadata["classification"]["predicted_label"]
        }
        
        # Store document
        doc_id = self.storage.add_document(
            content=content,
            metadata=metadata,
            source=metadata["filename"]
        )
        
        # Update with classification metadata
        classification_metadata = {
            "classification_confidence": metadata["classification"]["confidence"],
            "classification_scores": metadata["classification"]["all_scores"],
            "processing_job_id": metadata["job_id"]
        }
        
        self.storage.update_document_metadata(doc_id, classification_metadata)
        
        return doc_id
    
    async def _store_in_pinecone(self, content: str, metadata: Dict[str, Any]):
        """Store document embeddings in Pinecone"""
        # Simple chunking for now (could be enhanced)
        chunk_size = 1000
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        # Store each chunk
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_text": chunk[:200] + "..." if len(chunk) > 200 else chunk
            }
            
            # Store in Pinecone
            await self.pinecone_manager.upsert_document(
                doc_id=f"{metadata['job_id']}_chunk_{i}",
                content=chunk,
                metadata=chunk_metadata
            )

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
        classification_result = classifier.classify_text(text_content)
        
        # 4. Prepare enhanced metadata with deterministic labels
        metadata = {
            "filename": file_path.name,
            "file_path": str(file_path.absolute()),
            "processed_at": datetime.now().isoformat(),
            "processing_type": "unified_process_pdf",
            
            # Core classification metadata for filtering/analysis
            "predicted_label": classification_result["predicted_label"],
            "predicted_category": classification_result.get("predicted_category", classification_result["predicted_label"]),
            "predicted_confidence": classification_result["confidence"],
            
            # Detailed classification results
            "classification": {
                "predicted_label": classification_result["predicted_label"],
                "predicted_category": classification_result.get("predicted_category", classification_result["predicted_label"]),
                "confidence": classification_result["confidence"],
                "all_scores": classification_result.get("all_scores", {}),
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
                pinecone_manager = PineconeManager(config)
                
                # Simple chunking strategy
                chunk_size = 1000
                chunks = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]
                
                vectors_stored = 0
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_text": chunk[:200] + "..." if len(chunk) > 200 else chunk
                    }
                    
                    # Create unique ID for chunk
                    chunk_id = f"{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_chunk_{i}"
                    
                    # Store in Pinecone (sync version)
                    pinecone_manager.upsert_document(
                        doc_id=chunk_id,
                        content=chunk,
                        metadata=chunk_metadata
                    )
                    vectors_stored += 1
                
                storage_results["pinecone"] = {
                    "status": "success",
                    "chunks_stored": vectors_stored,
                    "total_chunks": len(chunks)
                }
                logger.info("Pinecone storage completed", vectors_stored=vectors_stored)
                
            except Exception as e:
                logger.error("Pinecone storage failed", error=str(e))
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
        
        # 7. Final result
        result = {
            "status": "success",
            "filename": file_path.name,
            "text_length": len(text_content),
            "classification": classification_result,
            "metadata": metadata,
            "storage": storage_results,
            "processing_time": (datetime.now() - datetime.fromisoformat(metadata["processed_at"].replace('Z', '+00:00'))).total_seconds()
        }
        
        logger.info("Unified PDF processing completed", 
                   filename=file_path.name,
                   classification=classification_result["predicted_label"],
                   confidence=classification_result["confidence"])
        
        return result
        
    except Exception as e:
        logger.error("Unified PDF processing failed", 
                    filename=file_path.name, 
                    error=str(e),
                    traceback=traceback.format_exc())
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
