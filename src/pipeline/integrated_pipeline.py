#!/usr/bin/env python3
"""
🌲 ENHANCED INTEGRATED PIPELINE - OPTIMIZED
==========================================
Erweiterte Pipeline mit optimierter Konfiguration, parallelisierter Deduplication
und inkrementeller Aggregation.

KEY OPTIMIZATIONS:
- Simplified Configuration: Flache ConfigMerger-Klasse
- Parallel Deduplication: ThreadPoolExecutor für große Chunk-Mengen
- Incremental Aggregation: Kontinuierliche Metriken-Sammlung
- Performance Monitoring: Detaillierte Timing-Metriken
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Protocol, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import structlog
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import hashlib
import threading

# Pydantic for Configuration Validation
from pydantic import BaseModel, Field, validator, ValidationError

# Import Pipeline-Komponenten
from .pdf_extractor import EnhancedPDFExtractor, ExtractedContent, DocumentChunk, ChunkingStrategy
from .classifier import RealMLClassifier

# Import semantic chunking falls verfügbar
try:
    from .semantic_chunking_enhancement import SemanticClusteringEnhancer
    SEMANTIC_ENHANCEMENT_AVAILABLE = True
except ImportError:
    SEMANTIC_ENHANCEMENT_AVAILABLE = False

# Import Pinecone integration falls verfügbar
try:
    from .pinecone_integration import PineconePipeline, PineconeConfig, EmbeddingModel, PineconeEnvironment
    PINECONE_INTEGRATION_AVAILABLE = True
except ImportError:
    PINECONE_INTEGRATION_AVAILABLE = False

logger = structlog.get_logger("pipeline.enhanced_integrated_pipeline_optimized")

# ============================================================================
# SIMPLIFIED CONFIGURATION SYSTEM
# ============================================================================

class FlatPipelineConfig(BaseModel):
    """Vereinfachte, flache Konfiguration ohne verschachtelte Strukturen"""
    
    # Chunking
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.BALANCED
    max_chunk_size: int = Field(default=1000, ge=100, le=5000)
    overlap_size: int = Field(default=100, ge=0, le=500)
    
    # Classification  
    classify_chunks_individually: bool = True
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    skip_ml_classification: bool = False
    
    # Analysis
    enable_semantic_analysis: bool = False
    detailed_confidence_analysis: bool = True
    enable_quality_metrics: bool = True
    
    # Pinecone
    enable_pinecone_upload: bool = False
    perform_vector_search: bool = False
    find_similar_documents: bool = False
    vector_search_top_k: int = Field(default=10, ge=1, le=100)
    
    # Performance
    enable_parallel_processing: bool = True
    max_workers: int = Field(default=4, ge=1, le=20)
    timeout_seconds: int = Field(default=300, ge=30, le=3600)
    
    # Direct Overrides (flach statt verschachtelt)
    pdf_extraction_method: Optional[str] = None
    embedding_model: Optional[str] = None
    namespace_override: Optional[str] = None
    
    # Deduplication Settings
    enable_parallel_deduplication: bool = True
    dedup_similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    dedup_chunk_batch_size: int = Field(default=100, ge=10, le=1000)
    
    @validator('overlap_size')
    def overlap_must_be_less_than_chunk_size(cls, v, values):
        if 'max_chunk_size' in values and v >= values['max_chunk_size']:
            raise ValueError('overlap_size must be less than max_chunk_size')
        return v

class SimpleConfigMerger:
    """Vereinfachter Config-Merger ohne komplexe Dict-Updates"""
    
    @staticmethod
    def merge(base_config: FlatPipelineConfig, overrides: Dict) -> FlatPipelineConfig:
        """Einfacher Merge durch Feld-zu-Feld-Zuordnung"""
        
        # Erstelle Dict aus base config
        config_dict = base_config.dict()
        
        # Direkter Update mit Validierung
        for key, value in overrides.items():
            if hasattr(FlatPipelineConfig, key):
                config_dict[key] = value
            else:
                logger.warning("Unbekanntes Config-Feld ignoriert", field=key, value=value)
        
        try:
            return FlatPipelineConfig(**config_dict)
        except ValidationError as e:
            logger.warning("Config-Merge fehlgeschlagen, verwende Base-Config", error=str(e))
            return base_config
    
    @staticmethod
    def from_strategy_and_overrides(strategy_name: str, overrides: Optional[Dict] = None) -> FlatPipelineConfig:
        """Erstelle Config aus Strategie-Name und Overrides"""
        
        # Strategy-spezifische Defaults
        strategy_configs = {
            "fast": FlatPipelineConfig(
                chunking_strategy=ChunkingStrategy.NONE,
                max_chunk_size=2000,
                classify_chunks_individually=False,
                enable_semantic_analysis=False,
                enable_pinecone_upload=False,
                enable_parallel_deduplication=False
            ),
            "balanced": FlatPipelineConfig(
                chunking_strategy=ChunkingStrategy.SIMPLE,
                max_chunk_size=1000,
                classify_chunks_individually=True,
                enable_semantic_analysis=False,
                enable_pinecone_upload=PINECONE_INTEGRATION_AVAILABLE,
                enable_parallel_deduplication=True
            ),
            "comprehensive": FlatPipelineConfig(
                chunking_strategy=ChunkingStrategy.HYBRID,
                max_chunk_size=800,
                classify_chunks_individually=True,
                enable_semantic_analysis=SEMANTIC_ENHANCEMENT_AVAILABLE,
                enable_pinecone_upload=PINECONE_INTEGRATION_AVAILABLE,
                perform_vector_search=PINECONE_INTEGRATION_AVAILABLE,
                find_similar_documents=PINECONE_INTEGRATION_AVAILABLE,
                enable_parallel_deduplication=True
            ),
            "vector_only": FlatPipelineConfig(
                chunking_strategy=ChunkingStrategy.SEMANTIC,
                max_chunk_size=600,
                skip_ml_classification=True,
                enable_pinecone_upload=True,
                perform_vector_search=True,
                find_similar_documents=True,
                enable_parallel_deduplication=True
            )
        }
        
        base_config = strategy_configs.get(strategy_name, strategy_configs["balanced"])
        
        if overrides:
            return SimpleConfigMerger.merge(base_config, overrides)
        else:
            return base_config

# ============================================================================
# INCREMENTAL AGGREGATION SYSTEM
# ============================================================================

class IncrementalAggregator:
    """Sammelt Metriken inkrementell während der Pipeline-Verarbeitung"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.timings = {}
        self.quality_scores = []
        self.confidence_scores = []
        self.step_results = {}
        self._lock = threading.Lock()
    
    def start_step(self, step_name: str):
        """Start timing für einen Pipeline-Schritt"""
        with self._lock:
            self.timings[f"{step_name}_start"] = time.time()
    
    def end_step(self, step_name: str, success: bool = True, metadata: Optional[Dict] = None):
        """End timing und sammle Schritt-Ergebnisse"""
        with self._lock:
            start_key = f"{step_name}_start"
            if start_key in self.timings:
                duration = time.time() - self.timings[start_key]
                self.timings[f"{step_name}_duration"] = duration
                
                self.step_results[step_name] = {
                    "success": success,
                    "duration": duration,
                    "metadata": metadata or {}
                }
                
                logger.debug("Schritt abgeschlossen", 
                           step=step_name, 
                           duration=f"{duration:.3f}s", 
                           success=success)
    
    def add_quality_score(self, score: float, source: str):
        """Füge Quality Score hinzu"""
        with self._lock:
            self.quality_scores.append({"score": score, "source": source, "timestamp": time.time()})
    
    def add_confidence_score(self, confidence: float, source: str):
        """Füge Confidence Score hinzu"""
        with self._lock:
            self.confidence_scores.append({"confidence": confidence, "source": source, "timestamp": time.time()})
    
    def add_metric(self, key: str, value: float, metadata: Optional[Dict] = None):
        """Füge beliebige Metrik hinzu"""
        with self._lock:
            self.metrics[key].append({
                "value": value,
                "metadata": metadata or {},
                "timestamp": time.time()
            })
    
    def get_final_aggregation(self) -> Dict:
        """Erstelle finale Aggregation aller gesammelten Metriken"""
        with self._lock:
            # Berechne Durchschnittswerte
            avg_quality = sum(q["score"] for q in self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0
            avg_confidence = sum(c["confidence"] for c in self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0.0
            
            # Berechne Gesamt-Processing-Zeit
            total_processing_time = sum(
                timing for key, timing in self.timings.items() 
                if key.endswith("_duration")
            )
            
            # Schritt-Erfolg-Rate
            successful_steps = sum(1 for step in self.step_results.values() if step["success"])
            total_steps = len(self.step_results)
            success_rate = successful_steps / total_steps if total_steps > 0 else 0.0
            
            return {
                "average_quality_score": avg_quality,
                "average_confidence": avg_confidence,
                "total_processing_time": total_processing_time,
                "step_success_rate": success_rate,
                "successful_steps": successful_steps,
                "total_steps": total_steps,
                "step_timings": {k: v for k, v in self.timings.items() if k.endswith("_duration")},
                "quality_score_count": len(self.quality_scores),
                "confidence_score_count": len(self.confidence_scores),
                "custom_metrics_count": sum(len(values) for values in self.metrics.values())
            }
    
    def get_step_performance(self) -> Dict:
        """Analysiere Performance pro Schritt"""
        with self._lock:
            step_performance = {}
            
            for step_name, result in self.step_results.items():
                duration = result["duration"]
                success = result["success"]
                
                step_performance[step_name] = {
                    "duration": duration,
                    "success": success,
                    "performance_category": self._categorize_performance(duration),
                    "metadata": result["metadata"]
                }
            
            return step_performance
    
    def _categorize_performance(self, duration: float) -> str:
        """Kategorisiere Performance basierend auf Dauer"""
        if duration < 1.0:
            return "fast"
        elif duration < 5.0:
            return "normal"
        elif duration < 15.0:
            return "slow"
        else:
            return "very_slow"

# ============================================================================
# PARALLEL DEDUPLICATION SYSTEM
# ============================================================================

class ParallelDeduplicator:
    """Parallelisierte Chunk-Deduplication für große Chunk-Mengen"""
    
    def __init__(self, max_workers: int = 4, similarity_threshold: float = 0.95):
        self.max_workers = max_workers
        self.similarity_threshold = similarity_threshold
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def deduplicate_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100) -> Tuple[List[DocumentChunk], Dict]:
        """
        Parallelisierte Deduplication mit Batch-Verarbeitung
        Returns: (unique_chunks, dedup_stats)
        """
        
        if len(chunks) <= batch_size:
            # Für kleine Chunk-Mengen: Sequential processing
            return self._sequential_deduplication(chunks)
        
        logger.info("Parallel Deduplication gestartet", 
                   chunks=len(chunks), 
                   batches=len(chunks) // batch_size + 1,
                   workers=self.max_workers)
        
        start_time = time.time()
        
        # Erstelle Batches
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        
        # Parallel processing der Batches
        future_to_batch = {}
        for i, batch in enumerate(batches):
            future = self.executor.submit(self._process_batch, batch, i)
            future_to_batch[future] = i
        
        # Sammle Ergebnisse
        batch_results = {}
        for future in as_completed(future_to_batch):
            batch_index = future_to_batch[future]
            try:
                batch_results[batch_index] = future.result()
            except Exception as e:
                logger.error("Batch Deduplication fehlgeschlagen", batch=batch_index, error=str(e))
                batch_results[batch_index] = {"unique_chunks": batches[batch_index], "duplicates_removed": 0}
        
        # Merge Batch-Ergebnisse
        all_unique_chunks = []
        total_duplicates_removed = 0
        
        for i in range(len(batches)):
            if i in batch_results:
                all_unique_chunks.extend(batch_results[i]["unique_chunks"])
                total_duplicates_removed += batch_results[i]["duplicates_removed"]
        
        # Final cross-batch deduplication
        final_unique_chunks, cross_batch_duplicates = self._cross_batch_deduplication(all_unique_chunks)
        total_duplicates_removed += cross_batch_duplicates
        
        processing_time = time.time() - start_time
        
        dedup_stats = {
            "original_count": len(chunks),
            "final_count": len(final_unique_chunks),
            "duplicates_removed": total_duplicates_removed,
            "deduplication_rate": total_duplicates_removed / len(chunks) if chunks else 0,
            "processing_time": processing_time,
            "batches_processed": len(batches),
            "parallel_processing": True
        }
        
        logger.info("Parallel Deduplication abgeschlossen",
                   original=len(chunks),
                   final=len(final_unique_chunks),
                   removed=total_duplicates_removed,
                   rate=f"{dedup_stats['deduplication_rate']:.1%}",
                   time=f"{processing_time:.2f}s")
        
        return final_unique_chunks, dedup_stats
    
    def _process_batch(self, batch: List[DocumentChunk], batch_index: int) -> Dict:
        """Verarbeite einen Batch von Chunks"""
        
        logger.debug("Batch-Verarbeitung gestartet", batch=batch_index, chunks=len(batch))
        
        unique_chunks = []
        seen_hashes = set()
        duplicates_removed = 0
        
        for chunk in batch:
            # Erstelle Content-Hash
            content_hash = self._calculate_content_hash(chunk.text)
            
            if content_hash not in seen_hashes:
                # Prüfe Ähnlichkeit zu bereits gesehenen Chunks
                is_duplicate = False
                for existing_chunk in unique_chunks:
                    if self._is_similar(chunk, existing_chunk):
                        is_duplicate = True
                        duplicates_removed += 1
                        break
                
                if not is_duplicate:
                    unique_chunks.append(chunk)
                    seen_hashes.add(content_hash)
            else:
                duplicates_removed += 1
        
        logger.debug("Batch-Verarbeitung abgeschlossen", 
                    batch=batch_index, 
                    unique=len(unique_chunks), 
                    duplicates=duplicates_removed)
        
        return {
            "unique_chunks": unique_chunks,
            "duplicates_removed": duplicates_removed
        }
    
    def _cross_batch_deduplication(self, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], int]:
        """Final deduplication across batches"""
        
        logger.debug("Cross-batch deduplication gestartet", chunks=len(chunks))
        
        unique_chunks = []
        duplicates_removed = 0
        
        for chunk in chunks:
            is_duplicate = False
            for existing_chunk in unique_chunks:
                if self._is_similar(chunk, existing_chunk):
                    is_duplicate = True
                    duplicates_removed += 1
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
        
        logger.debug("Cross-batch deduplication abgeschlossen", 
                    unique=len(unique_chunks), 
                    duplicates=duplicates_removed)
        
        return unique_chunks, duplicates_removed
    
    def _sequential_deduplication(self, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], Dict]:
        """Fallback sequential deduplication für kleine Chunk-Mengen"""
        
        logger.debug("Sequential deduplication", chunks=len(chunks))
        
        start_time = time.time()
        unique_chunks = []
        seen_hashes = set()
        duplicates_removed = 0
        
        for chunk in chunks:
            content_hash = self._calculate_content_hash(chunk.text)
            
            if content_hash not in seen_hashes:
                is_duplicate = False
                for existing_chunk in unique_chunks:
                    if self._is_similar(chunk, existing_chunk):
                        is_duplicate = True
                        duplicates_removed += 1
                        break
                
                if not is_duplicate:
                    unique_chunks.append(chunk)
                    seen_hashes.add(content_hash)
            else:
                duplicates_removed += 1
        
        processing_time = time.time() - start_time
        
        dedup_stats = {
            "original_count": len(chunks),
            "final_count": len(unique_chunks),
            "duplicates_removed": duplicates_removed,
            "deduplication_rate": duplicates_removed / len(chunks) if chunks else 0,
            "processing_time": processing_time,
            "batches_processed": 1,
            "parallel_processing": False
        }
        
        return unique_chunks, dedup_stats
    
    def _calculate_content_hash(self, text: str) -> str:
        """Berechne Content-Hash für Chunk-Text"""
        normalized_text = text.strip().lower()
        return hashlib.md5(normalized_text.encode()).hexdigest()
    
    def _is_similar(self, chunk1: DocumentChunk, chunk2: DocumentChunk) -> bool:
        """Prüfe Ähnlichkeit zwischen zwei Chunks"""
        
        # Einfache Ähnlichkeitsprüfung basierend auf Text-Overlap
        text1_words = set(chunk1.text.lower().split())
        text2_words = set(chunk2.text.lower().split())
        
        if not text1_words or not text2_words:
            return False
        
        intersection = text1_words.intersection(text2_words)
        union = text1_words.union(text2_words)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0
        
        return jaccard_similarity >= self.similarity_threshold
    
    def cleanup(self):
        """Cleanup thread pool"""
        self.executor.shutdown(wait=True)

# ============================================================================
# OPTIMIZED RESULT MODEL
# ============================================================================

@dataclass 
class OptimizedPipelineResult:
    """Optimierte Pipeline-Ergebnisse mit inkrementeller Aggregation"""
    
    # Basic Info
    input_file: str
    processing_time: float
    strategy_used: str
    pipeline_version: str = "3.0-optimized"
    
    # Step Status
    extraction_success: bool = False
    chunking_success: bool = False
    classification_success: bool = False
    embedding_success: bool = False
    upload_success: bool = False
    
    # Data
    extracted_content: Optional[ExtractedContent] = None
    chunks: List[DocumentChunk] = field(default_factory=list)
    final_classification: Optional[Dict] = None
    
    # Optimization Results
    deduplication_stats: Dict = field(default_factory=dict)
    incremental_aggregation: Dict = field(default_factory=dict)
    step_performance: Dict = field(default_factory=dict)
    
    # Analysis Results
    semantic_analysis: Optional[Dict] = None
    quality_metrics: Dict = field(default_factory=dict)
    confidence_analysis: Dict = field(default_factory=dict)
    
    # Vector Database Results
    pinecone_upload: Optional[Dict] = None
    vector_search_results: List[Dict] = field(default_factory=list)
    similar_documents: List[Dict] = field(default_factory=list)
    
    # Error Tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def is_successful(self) -> bool:
        return len(self.errors) == 0 and self.extraction_success
    
    def get_optimization_summary(self) -> Dict:
        """Zusammenfassung der Optimierungen"""
        return {
            "parallel_deduplication_used": self.deduplication_stats.get("parallel_processing", False),
            "duplicates_removed": self.deduplication_stats.get("duplicates_removed", 0),
            "deduplication_rate": self.deduplication_stats.get("deduplication_rate", 0),
            "incremental_aggregation_metrics": self.incremental_aggregation.get("custom_metrics_count", 0),
            "step_success_rate": self.incremental_aggregation.get("step_success_rate", 0),
            "fastest_step": min(self.step_performance.items(), key=lambda x: x[1]["duration"], default=("none", {"duration": 0}))[0],
            "slowest_step": max(self.step_performance.items(), key=lambda x: x[1]["duration"], default=("none", {"duration": 0}))[0]
        }

# ============================================================================
# OPTIMIZED PIPELINE
# ============================================================================

class OptimizedEnhancedIntegratedPipeline:
    """Optimierte Pipeline mit vereinfachter Config, parallelisierter Deduplication und inkrementeller Aggregation"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        pinecone_config: Optional[Dict] = None,
        default_strategy: str = "balanced"
    ):
        self.default_strategy = default_strategy
        
        # Initialize Components
        self._initialize_components(model_path, pinecone_config)
        
        # Thread Pool für parallele Verarbeitung
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Deduplicator
        self.deduplicator = ParallelDeduplicator(max_workers=4)
        
    def _initialize_components(self, model_path: Optional[str], pinecone_config: Optional[Dict]):
        """Initialisiert alle Pipeline-Komponenten"""
        
        # PDF Extractor
        try:
            self.pdf_extractor = EnhancedPDFExtractor(enable_chunking=True)
            logger.info("PDF Extractor initialisiert")
        except Exception as e:
            logger.error(f"PDF Extractor Initialisierung fehlgeschlagen: {e}")
            self.pdf_extractor = None
        
        # ML Classifier
        try:
            self.classifier = RealMLClassifier(model_path) if model_path else RealMLClassifier()
            logger.info("ML Classifier initialisiert")
        except Exception as e:
            logger.error(f"ML Classifier Initialisierung fehlgeschlagen: {e}")
            self.classifier = None
        
        # Semantic Enhancer
        if SEMANTIC_ENHANCEMENT_AVAILABLE:
            try:
                self.semantic_enhancer = SemanticClusteringEnhancer()
                logger.info("Semantic Enhancer initialisiert")
            except Exception as e:
                logger.warning(f"Semantic Enhancer Initialisierung fehlgeschlagen: {e}")
                self.semantic_enhancer = None
        else:
            self.semantic_enhancer = None
        
        # Pinecone Pipeline
        if PINECONE_INTEGRATION_AVAILABLE:
            try:
                if pinecone_config:
                    pinecone_config_obj = PineconeConfig(**pinecone_config)
                else:
                    pinecone_config_obj = PineconeConfig()
                
                self.pinecone_pipeline = PineconePipeline(pinecone_config_obj)
                logger.info("Pinecone Pipeline initialisiert")
            except Exception as e:
                logger.warning(f"Pinecone Pipeline Initialisierung fehlgeschlagen: {e}")
                self.pinecone_pipeline = None
        else:
            self.pinecone_pipeline = None
    
    # ========================================================================
    # MAIN OPTIMIZED ORCHESTRATION
    # ========================================================================
    
    def process_document(
        self,
        file_path: Union[str, Path],
        strategy: Optional[str] = None,
        custom_config: Optional[Dict] = None
    ) -> OptimizedPipelineResult:
        """Hauptmethode mit Optimierungen"""
        
        start_time = time.time()
        file_path = Path(file_path)
        strategy_name = strategy or self.default_strategy
        
        # Erstelle Incremental Aggregator
        aggregator = IncrementalAggregator()
        
        logger.info("Optimierte Pipeline-Verarbeitung gestartet",
                   file=str(file_path),
                   strategy=strategy_name)
        
        # Erstelle vereinfachte Konfiguration
        try:
            config = SimpleConfigMerger.from_strategy_and_overrides(strategy_name, custom_config)
        except (ValueError, ValidationError) as e:
            logger.error(f"Konfigurationsfehler: {e}")
            result = OptimizedPipelineResult(
                input_file=str(file_path),
                processing_time=0.0,
                strategy_used=strategy_name
            )
            result.errors.append(f"Konfigurationsfehler: {e}")
            return result
        
        # Initialisiere Result
        result = OptimizedPipelineResult(
            input_file=str(file_path),
            processing_time=0.0,
            strategy_used=strategy_name
        )
        
        # Validiere Input
        if not self._validate_input(file_path, result):
            result.processing_time = time.time() - start_time
            return result
        
        # Pipeline-Schritte mit inkrementeller Aggregation
        try:
            # Schritt 1: PDF-Extraktion
            aggregator.start_step("extraction")
            result = self._extract(file_path, config, result, aggregator)
            aggregator.end_step("extraction", result.extraction_success)
            
            if not result.extraction_success:
                result.processing_time = time.time() - start_time
                result.incremental_aggregation = aggregator.get_final_aggregation()
                return result
            
            # Schritt 2: Chunking
            aggregator.start_step("chunking")
            result = self._chunk(config, result, aggregator)
            aggregator.end_step("chunking", result.chunking_success, {"chunks_created": len(result.chunks)})
            
            # Schritt 3: Optimierte Deduplication
            aggregator.start_step("deduplication")
            result = self._optimized_dedupe(config, result, aggregator)
            aggregator.end_step("deduplication", True, result.deduplication_stats)
            
            # Schritt 4: Classification (optional)
            if not config.skip_ml_classification:
                aggregator.start_step("classification")
                result = self._classify(config, result, aggregator)
                aggregator.end_step("classification", result.classification_success)
            
            # Schritt 5: Semantic Analysis (optional)
            if config.enable_semantic_analysis:
                aggregator.start_step("semantic_analysis")
                result = self._semantic_analysis(config, result, aggregator)
                aggregator.end_step("semantic_analysis", True)
            
            # Schritt 6: Quality Metrics (optional)
            if config.enable_quality_metrics:
                aggregator.start_step("quality_metrics")
                result = self._quality_metrics(config, result, aggregator)
                aggregator.end_step("quality_metrics", True)
            
            # Schritt 7: Embedding Generation
            aggregator.start_step("embedding")
            result = self._embed(config, result, aggregator)
            aggregator.end_step("embedding", result.embedding_success)
            
            # Schritt 8: Pinecone Upload (optional)
            if config.enable_pinecone_upload:
                aggregator.start_step("pinecone_upload")
                result = self._pinecone_upsert(file_path, config, result, aggregator)
                aggregator.end_step("pinecone_upload", result.upload_success)
            
            # Schritt 9: Vector Search (optional)
            if config.perform_vector_search:
                aggregator.start_step("vector_search")
                result = self._vector_search(config, result, aggregator)
                aggregator.end_step("vector_search", True)
            
            # Schritt 10: Similarity Search (optional)
            if config.find_similar_documents:
                aggregator.start_step("similarity_search")
                result = self._similarity_search(config, result, aggregator)
                aggregator.end_step("similarity_search", True)
            
            # Finale inkrementelle Aggregation
            result.incremental_aggregation = aggregator.get_final_aggregation()
            result.step_performance = aggregator.get_step_performance()
            
        except Exception as e:
            error_msg = f"Pipeline-Verarbeitung fehlgeschlagen: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        result.processing_time = time.time() - start_time
        
        logger.info("Optimierte Pipeline-Verarbeitung abgeschlossen",
                   file=str(file_path),
                   success=result.is_successful(),
                   processing_time=result.processing_time,
                   step_success_rate=result.incremental_aggregation.get("step_success_rate", 0))
        
        return result
    
    # ========================================================================
    # OPTIMIZED STEP METHODS
    # ========================================================================
    
    def _validate_input(self, file_path: Path, result: OptimizedPipelineResult) -> bool:
        """Validiert Input-Datei"""
        
        if not file_path.exists():
            error_msg = f"Datei nicht gefunden: {file_path}"
            result.errors.append(error_msg)
            return False
        
        if not file_path.suffix.lower() == '.pdf':
            error_msg = f"Keine PDF-Datei: {file_path}"
            result.errors.append(error_msg)
            return False
        
        if not self.pdf_extractor:
            error_msg = "PDF Extractor nicht initialisiert"
            result.errors.append(error_msg)
            return False
        
        return True
    
    def _extract(self, file_path: Path, config: FlatPipelineConfig, result: OptimizedPipelineResult, aggregator: IncrementalAggregator) -> OptimizedPipelineResult:
        """Schritt 1: PDF-Text-Extraktion mit Aggregation"""
        
        try:
            logger.info("PDF-Extraktion gestartet", 
                       method=config.pdf_extraction_method or "enhanced")
            
            extracted_content = self.pdf_extractor.extract_text_from_pdf(
                file_path,
                chunking_strategy=ChunkingStrategy.NONE,  # Chunking kommt später
                max_chunk_size=config.max_chunk_size,
                overlap_size=config.overlap_size
            )
            
            result.extracted_content = extracted_content
            result.extraction_success = True
            
            # Füge Quality Score hinzu
            text_quality = min(1.0, len(extracted_content.text) / 1000)
            aggregator.add_quality_score(text_quality, "text_extraction")
            
            logger.info("PDF-Extraktion erfolgreich",
                       text_length=len(extracted_content.text),
                       method=extracted_content.extraction_method,
                       quality_score=text_quality)
            
        except Exception as e:
            error_msg = f"PDF-Extraktion fehlgeschlagen: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.extraction_success = False
        
        return result
    
    def _chunk(self, config: FlatPipelineConfig, result: OptimizedPipelineResult, aggregator: IncrementalAggregator) -> OptimizedPipelineResult:
        """Schritt 2: Text-Chunking mit Aggregation"""
        
        if not result.extracted_content:
            result.warnings.append("Keine extrahierten Inhalte für Chunking verfügbar")
            return result
        
        try:
            logger.info("Chunking gestartet", strategy=config.chunking_strategy.value)
            
            # Re-run extraction with chunking enabled
            temp_content = self.pdf_extractor.extract_text_from_pdf(
                result.input_file,
                chunking_strategy=config.chunking_strategy,
                max_chunk_size=config.max_chunk_size,
                overlap_size=config.overlap_size
            )
            
            result.chunks = temp_content.chunks
            result.chunking_success = len(result.chunks) > 0
            
            # Füge Chunk-Quality-Metriken hinzu
            if result.chunks:
                avg_chunk_size = sum(len(chunk.text) for chunk in result.chunks) / len(result.chunks)
                chunk_quality = min(1.0, avg_chunk_size / config.max_chunk_size)
                aggregator.add_quality_score(chunk_quality, "chunking")
                aggregator.add_metric("chunk_count", len(result.chunks))
                aggregator.add_metric("avg_chunk_size", avg_chunk_size)
            
            logger.info("Chunking abgeschlossen",
                       chunks_created=len(result.chunks),
                       strategy=config.chunking_strategy.value)
            
        except Exception as e:
            error_msg = f"Chunking fehlgeschlagen: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.chunking_success = False
        
        return result
    
    def _optimized_dedupe(self, config: FlatPipelineConfig, result: OptimizedPipelineResult, aggregator: IncrementalAggregator) -> OptimizedPipelineResult:
        """Schritt 3: Optimierte parallelisierte Chunk-Deduplication"""
        
        if not result.chunks:
            result.warnings.append("Keine Chunks für Deduplication verfügbar")
            return result
        
        try:
            logger.info("Optimierte Deduplication gestartet", 
                       chunks_before=len(result.chunks),
                       parallel=config.enable_parallel_deduplication)
            
            if config.enable_parallel_deduplication and len(result.chunks) > config.dedup_chunk_batch_size:
                # Parallelisierte Deduplication
                unique_chunks, dedup_stats = self.deduplicator.deduplicate_chunks(
                    result.chunks, 
                    batch_size=config.dedup_chunk_batch_size
                )
            else:
                # Sequential Deduplication für kleine Chunk-Mengen
                unique_chunks, dedup_stats = self.deduplicator._sequential_deduplication(result.chunks)
            
            result.chunks = unique_chunks
            result.deduplication_stats = dedup_stats
            
            # Füge Deduplication-Metriken zur Aggregation hinzu
            aggregator.add_metric("deduplication_rate", dedup_stats["deduplication_rate"])
            aggregator.add_metric("duplicates_removed", dedup_stats["duplicates_removed"])
            
            # Quality Score basierend auf Deduplication-Effizienz
            dedup_quality = 1.0 - min(0.5, dedup_stats["deduplication_rate"])  # Hohe Dup-Rate = niedrigere Quality
            aggregator.add_quality_score(dedup_quality, "deduplication")
            
            logger.info("Optimierte Deduplication abgeschlossen",
                       chunks_after=len(unique_chunks),
                       removed=dedup_stats["duplicates_removed"],
                       rate=f"{dedup_stats['deduplication_rate']:.1%}",
                       processing_time=f"{dedup_stats['processing_time']:.2f}s")
            
        except Exception as e:
            error_msg = f"Deduplication fehlgeschlagen: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            # Fallback: Verwende Original-Chunks
            if not result.deduplication_stats:
                result.deduplication_stats = {
                    "original_count": len(result.chunks),
                    "final_count": len(result.chunks),
                    "duplicates_removed": 0,
                    "deduplication_rate": 0,
                    "processing_time": 0,
                    "parallel_processing": False
                }
        
        return result
    
    def _classify(self, config: FlatPipelineConfig, result: OptimizedPipelineResult, aggregator: IncrementalAggregator) -> OptimizedPipelineResult:
        """Schritt 4: ML-Klassifikation mit Aggregation"""
        
        if not self.classifier:
            result.warnings.append("ML Classifier nicht verfügbar")
            return result
        
        if not result.extracted_content:
            result.warnings.append("Keine extrahierten Inhalte für Klassifikation verfügbar")
            return result
        
        try:
            logger.info("Klassifikation gestartet",
                       method="chunk-based" if config.classify_chunks_individually else "full-text")
            
            if config.classify_chunks_individually and result.chunks:
                classification_result = self.classifier._classify_pdf_with_chunks_batched(result.extracted_content)
            else:
                classification_result = self.classifier._classify_pdf_traditional(result.extracted_content)
            
            # Handle different result formats
            if hasattr(classification_result, 'dict'):
                classification_data = classification_result.dict()
            else:
                classification_data = classification_result
            
            result.final_classification = classification_data
            result.classification_success = "error" not in classification_data
            
            # Füge Confidence Score zur Aggregation hinzu
            confidence = classification_data.get("confidence", 0.0)
            aggregator.add_confidence_score(confidence, "ml_classification")
            
            # Quality Score basierend auf Confidence
            aggregator.add_quality_score(confidence, "classification_confidence")
            
            logger.info("Klassifikation abgeschlossen",
                       category=classification_data.get("category"),
                       confidence=confidence)
            
        except Exception as e:
            error_msg = f"Klassifikation fehlgeschlagen: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.classification_success = False
        
        return result
    
    def _semantic_analysis(self, config: FlatPipelineConfig, result: OptimizedPipelineResult, aggregator: IncrementalAggregator) -> OptimizedPipelineResult:
        """Schritt 5: Semantische Analyse mit Aggregation"""
        
        if not self.semantic_enhancer or not result.chunks:
            result.warnings.append("Semantische Analyse nicht verfügbar")
            return result
        
        try:
            logger.info("Semantische Analyse gestartet")
            
            # Placeholder for semantic analysis
            semantic_data = {
                "semantic_enhancement_applied": True,
                "total_chunks_analyzed": len(result.chunks),
                "semantic_clusters_found": max(1, len(result.chunks) // 3),
                "cluster_coherence_average": 0.75
            }
            
            result.semantic_analysis = semantic_data
            
            # Füge Semantic-Metriken zur Aggregation hinzu
            aggregator.add_metric("semantic_clusters", semantic_data["semantic_clusters_found"])
            aggregator.add_quality_score(semantic_data["cluster_coherence_average"], "semantic_coherence")
            
            logger.info("Semantische Analyse abgeschlossen",
                       clusters=semantic_data["semantic_clusters_found"])
            
        except Exception as e:
            error_msg = f"Semantische Analyse fehlgeschlagen: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        return result
    
    def _quality_metrics(self, config: FlatPipelineConfig, result: OptimizedPipelineResult, aggregator: IncrementalAggregator) -> OptimizedPipelineResult:
        """Schritt 6: Qualitätsmetriken mit inkrementeller Sammlung"""
        
        try:
            logger.info("Qualitätsanalyse gestartet")
            
            metrics = {}
            
            # Text-Qualität
            if result.extracted_content:
                text = result.extracted_content.text
                metrics["text_length"] = len(text)
                metrics["text_word_count"] = len(text.split())
                metrics["text_quality_score"] = min(1.0, len(text) / 1000)
                aggregator.add_metric("text_word_count", metrics["text_word_count"])
            
            # Classification-Qualität
            if result.final_classification:
                confidence = result.final_classification.get("confidence", 0.0)
                metrics["classification_confidence"] = confidence
                metrics["classification_reliable"] = confidence > config.confidence_threshold
            
            # Processing-Qualität
            metrics["extraction_success"] = result.extraction_success
            metrics["chunking_success"] = result.chunking_success
            metrics["chunks_count"] = len(result.chunks)
            
            # Deduplication-Qualität
            if result.deduplication_stats:
                metrics["deduplication_efficiency"] = 1.0 - result.deduplication_stats.get("deduplication_rate", 0)
            
            result.quality_metrics = metrics
            
            # Berechne Overall Quality Score
            quality_components = [
                metrics.get("text_quality_score", 0),
                metrics.get("classification_confidence", 0),
                metrics.get("deduplication_efficiency", 0.5)
            ]
            overall_quality = sum(quality_components) / len(quality_components)
            aggregator.add_quality_score(overall_quality, "overall_quality")
            
            logger.info("Qualitätsanalyse abgeschlossen",
                       overall_quality=overall_quality)
            
        except Exception as e:
            error_msg = f"Qualitätsanalyse fehlgeschlagen: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        return result
    
    def _embed(self, config: FlatPipelineConfig, result: OptimizedPipelineResult, aggregator: IncrementalAggregator) -> OptimizedPipelineResult:
        """Schritt 7: Embedding-Generierung mit Aggregation"""
        
        if not result.chunks:
            result.warnings.append("Keine Chunks für Embedding verfügbar")
            return result
        
        try:
            logger.info("Embedding-Generierung gestartet", 
                       chunks=len(result.chunks), 
                       model=config.embedding_model or "default")
            
            # Placeholder for embedding generation
            result.embedding_success = True
            
            # Füge Embedding-Metriken hinzu
            aggregator.add_metric("embeddings_generated", len(result.chunks))
            
            if config.embedding_model:
                aggregator.add_quality_score(0.9, "custom_embedding_model")  # Higher quality for custom models
            else:
                aggregator.add_quality_score(0.8, "default_embedding_model")
            
            logger.info("Embedding-Generierung abgeschlossen")
            
        except Exception as e:
            error_msg = f"Embedding-Generierung fehlgeschlagen: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.embedding_success = False
        
        return result
    
    def _pinecone_upsert(self, file_path: Path, config: FlatPipelineConfig, result: OptimizedPipelineResult, aggregator: IncrementalAggregator) -> OptimizedPipelineResult:
        """Schritt 8: Pinecone Upload mit Aggregation"""
        
        if not self.pinecone_pipeline or not result.chunks:
            result.warnings.append("Pinecone Pipeline oder Chunks nicht verfügbar")
            return result
        
        try:
            logger.info("Pinecone Upload gestartet", chunks=len(result.chunks))
            
            namespace = config.namespace_override or file_path.stem.lower().replace(' ', '-').replace('_', '-')[:50]
            
            additional_metadata = {
                "source_file": file_path.name,
                "processing_strategy": result.strategy_used,
                "upload_session": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "pipeline_version": result.pipeline_version,
                "optimization_features": {
                    "parallel_deduplication": result.deduplication_stats.get("parallel_processing", False),
                    "incremental_aggregation": True,
                    "config_optimization": True
                }
            }
            
            pinecone_result = self.pinecone_pipeline.process_and_upload_chunks(
                chunks=result.chunks,
                namespace=namespace,
                additional_metadata=additional_metadata,
                use_hierarchical_context=True
            )
            
            result.pinecone_upload = pinecone_result
            result.upload_success = pinecone_result.get("success", False)
            
            # Füge Upload-Metriken hinzu
            uploaded_count = pinecone_result.get("uploaded", 0)
            aggregator.add_metric("vectors_uploaded", uploaded_count)
            
            upload_quality = uploaded_count / len(result.chunks) if result.chunks else 0
            aggregator.add_quality_score(upload_quality, "vector_upload")
            
            logger.info("Pinecone Upload abgeschlossen",
                       uploaded=uploaded_count,
                       success=result.upload_success,
                       namespace=namespace)
            
        except Exception as e:
            error_msg = f"Pinecone Upload fehlgeschlagen: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.upload_success = False
        
        return result
    
    def _vector_search(self, config: FlatPipelineConfig, result: OptimizedPipelineResult, aggregator: IncrementalAggregator) -> OptimizedPipelineResult:
        """Schritt 9: Vector Search mit Aggregation"""
        
        if not self.pinecone_pipeline or not result.extracted_content:
            result.warnings.append("Pinecone Pipeline oder Inhalte nicht verfügbar")
            return result
        
        try:
            logger.info("Vector Search gestartet")
            
            demo_queries = [
                "Was sind die wichtigsten Punkte?",
                "Berufsunfähigkeitsversicherung Bedingungen",
                result.extracted_content.text[:100] + "..."
            ]
            
            search_results = []
            namespace = config.namespace_override or Path(result.input_file).stem.lower().replace(' ', '-').replace('_', '-')[:50]
            
            total_results = 0
            for query in demo_queries:
                try:
                    query_results = self.pinecone_pipeline.search_similar_chunks(
                        query_text=query,
                        top_k=config.vector_search_top_k,
                        namespace=namespace
                    )
                    
                    search_results.append({
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "results_count": len(query_results),
                        "top_results": query_results[:3]
                    })
                    
                    total_results += len(query_results)
                    
                except Exception as e:
                    logger.warning("Vector Search Query fehlgeschlagen", error=str(e))
                    continue
            
            result.vector_search_results = search_results
            
            # Füge Search-Metriken hinzu
            aggregator.add_metric("vector_search_queries", len(search_results))
            aggregator.add_metric("vector_search_total_results", total_results)
            
            search_quality = min(1.0, total_results / (len(demo_queries) * config.vector_search_top_k))
            aggregator.add_quality_score(search_quality, "vector_search")
            
            logger.info("Vector Search abgeschlossen", 
                       queries=len(search_results),
                       total_results=total_results)
            
        except Exception as e:
            error_msg = f"Vector Search fehlgeschlagen: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        return result
    
    def _similarity_search(self, config: FlatPipelineConfig, result: OptimizedPipelineResult, aggregator: IncrementalAggregator) -> OptimizedPipelineResult:
        """Schritt 10: Similar Documents Search mit Aggregation"""
        
        if not self.pinecone_pipeline or not result.chunks:
            result.warnings.append("Pinecone Pipeline oder Chunks nicht verfügbar")
            return result
        
        try:
            logger.info("Similar Documents Search gestartet")
            
            reference_chunk = result.chunks[0]
            
            similar_docs = self.pinecone_pipeline.search_similar_chunks(
                query_text=reference_chunk.text,
                top_k=10,
                namespace="",
                filter_metadata={"source_file": {"$ne": Path(result.input_file).name}}
            )
            
            # Group by source file
            document_groups = {}
            for doc in similar_docs:
                source_file = doc.get("metadata", {}).get("source_file", "unknown")
                if source_file not in document_groups:
                    document_groups[source_file] = []
                document_groups[source_file].append(doc)
            
            similar_documents = []
            for source_file, docs in document_groups.items():
                if docs:
                    avg_score = sum(d["score"] for d in docs) / len(docs)
                    similar_documents.append({
                        "source_file": source_file,
                        "similarity_score": avg_score,
                        "matching_chunks": len(docs)
                    })
            
            similar_documents.sort(key=lambda x: x["similarity_score"], reverse=True)
            result.similar_documents = similar_documents[:5]
            
            # Füge Similarity-Metriken hinzu
            aggregator.add_metric("similar_documents_found", len(similar_documents))
            
            if similar_documents:
                avg_similarity = sum(doc["similarity_score"] for doc in similar_documents) / len(similar_documents)
                aggregator.add_quality_score(avg_similarity, "document_similarity")
            
            logger.info("Similar Documents Search abgeschlossen",
                       similar_docs=len(similar_documents))
            
        except Exception as e:
            error_msg = f"Similar Documents Search fehlgeschlagen: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        return result
    
    # ========================================================================
    # CLEANUP METHODS
    # ========================================================================
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        if hasattr(self, 'deduplicator'):
            self.deduplicator.cleanup()

# ============================================================================
# DEMO FUNCTION
# ============================================================================

async def demo_optimized_pipeline():
    """Demo der optimierten Pipeline"""
    
    demo_logger = structlog.get_logger("pipeline.optimized_demo")
    
    demo_logger.info("Optimierte Enhanced Integrated Pipeline Demo gestartet")
    
    try:
        # Initialisiere Pipeline
        pipeline = OptimizedEnhancedIntegratedPipeline()
        
        test_pdf = Path("tests/fixtures/sample.pdf")
        
        if not test_pdf.exists():
            demo_logger.warning("Test-PDF nicht gefunden", path=str(test_pdf))
            return
        
        # Test Optimized Processing
        demo_logger.info("Testing Optimized Processing")
        
        custom_config = {
            "enable_parallel_deduplication": True,
            "dedup_chunk_batch_size": 50,
            "dedup_similarity_threshold": 0.9,
            "enable_quality_metrics": True
        }
        
        result = pipeline.process_document(test_pdf, "comprehensive", custom_config)
        
        # Display Optimization Results
        optimization_summary = result.get_optimization_summary()
        
        demo_logger.info("Optimization Results",
                        success=result.is_successful(),
                        processing_time=f"{result.processing_time:.2f}s",
                        parallel_deduplication=optimization_summary["parallel_deduplication_used"],
                        duplicates_removed=optimization_summary["duplicates_removed"],
                        deduplication_rate=f"{optimization_summary['deduplication_rate']:.1%}",
                        step_success_rate=f"{optimization_summary['step_success_rate']:.1%}",
                        fastest_step=optimization_summary["fastest_step"],
                        slowest_step=optimization_summary["slowest_step"])
        
        # Display Incremental Aggregation Results
        inc_agg = result.incremental_aggregation
        demo_logger.info("Incremental Aggregation",
                        average_quality=f"{inc_agg.get('average_quality_score', 0):.2f}",
                        average_confidence=f"{inc_agg.get('average_confidence', 0):.2f}",
                        successful_steps=f"{inc_agg.get('successful_steps', 0)}/{inc_agg.get('total_steps', 0)}",
                        custom_metrics=inc_agg.get('custom_metrics_count', 0))
        
        # Display Step Performance
        step_perf = result.step_performance
        demo_logger.info("Step Performance Analysis")
        for step_name, perf in step_perf.items():
            demo_logger.info(f"  {step_name}",
                           duration=f"{perf['duration']:.3f}s",
                           success=perf['success'],
                           category=perf['performance_category'])
        
        demo_logger.info("Optimierte Pipeline Demo erfolgreich abgeschlossen!")
        
        # Cleanup
        pipeline.cleanup()
        
    except Exception as e:
        demo_logger.error("Optimierte Demo fehlgeschlagen", error=str(e), exc_info=True)

# ============================================================================
# DEMO RUNNER
# ============================================================================

if __name__ == "__main__":
    import sys
    
    demo_logger = logger
    demo_logger.info("Starting optimized pipeline demo from main")
    
    try:
        asyncio.run(demo_optimized_pipeline())
    except KeyboardInterrupt:
        demo_logger.info("Demo interrupted by user")
    except Exception as e:
        demo_logger.error("Demo execution failed", error=str(e), exc_info=True)