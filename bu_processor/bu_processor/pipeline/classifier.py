"""ML-basierter Dokumentklassifizierer mit Retry-Mechanismus und Batch-Verarbeitung."""

import asyncio
import logging
import os
import random
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Pydantic für Schema-Validation
try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback BaseModel
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# Config import mit Fallback
try:
    from ..core.config import ML_MODEL_PATH, MAX_SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD, USE_GPU
except ImportError:  # pragma: no cover - final fallback
    # Fallback-Werte falls Config nicht gefunden
    ML_MODEL_PATH = "bert-base-german-cased"
    MAX_SEQUENCE_LENGTH = 512
    CONFIDENCE_THRESHOLD = 0.8
    USE_GPU = True

from .pdf_extractor import EnhancedPDFExtractor, ExtractedContent, DocumentChunk, ChunkingStrategy

logger = logging.getLogger(__name__)

# === PYDANTIC SCHEMA-MODELLE ===

if PYDANTIC_AVAILABLE:
    class ClassificationResult(BaseModel):
        """Typisierte Klassifikationsergebnisse mit Pydantic-Validation"""
        category: int = Field(..., ge=0, description="Predicted category (0-based)")
        confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
        is_confident: bool = Field(..., description="Whether confidence meets threshold")
        input_type: str = Field(..., description="Type of input processed")
        text_length: Optional[int] = Field(None, ge=0, description="Length of processed text")
        processing_time: Optional[float] = Field(None, ge=0.0, description="Processing time in seconds")
        model_version: Optional[str] = Field(None, description="Model version used")
        timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Processing timestamp")

        @validator('confidence')
        def validate_confidence(cls, v):
            if not 0.0 <= v <= 1.0:
                raise ValueError('Confidence must be between 0 and 1')
            return v

    class PDFClassificationResult(ClassificationResult):
        """Erweiterte PDF-spezifische Klassifikationsergebnisse"""
        file_path: str = Field(..., description="Path to processed PDF file")
        page_count: int = Field(..., ge=1, description="Number of pages in PDF")
        extraction_method: str = Field(..., description="PDF extraction method used")
        pdf_metadata: Optional[Dict[str, Any]] = Field(None, description="PDF metadata")
        chunking_enabled: bool = Field(default=False, description="Whether chunking was used")
        chunking_method: Optional[str] = Field(None, description="Chunking strategy applied")

    class BatchClassificationResult(BaseModel):
        """Batch-Klassifikationsergebnisse"""
        total_processed: int = Field(..., ge=0, description="Total items processed")
        successful: int = Field(..., ge=0, description="Successfully processed items")
        failed: int = Field(..., ge=0, description="Failed items")
        batch_time: float = Field(..., ge=0.0, description="Total batch processing time")
        results: List[ClassificationResult] = Field(..., description="Individual results")
        batch_id: Optional[str] = Field(None, description="Unique batch identifier")
        
        @validator('failed')
        def validate_failed_count(cls, v, values):
            if 'successful' in values and 'total_processed' in values:
                if v + values['successful'] != values['total_processed']:
                    raise ValueError('Failed + Successful must equal Total')
            return v

    class RetryStats(BaseModel):
        """Retry-Statistiken"""
        attempts: int = Field(..., ge=1, description="Number of attempts made")
        total_delay: float = Field(..., ge=0.0, description="Total delay time")
        final_success: bool = Field(..., description="Whether final attempt succeeded")
        error_types: List[str] = Field(default_factory=list, description="Types of errors encountered")

else:
    # Fallback für Pydantic-freie Umgebungen
    ClassificationResult = BaseModel
    PDFClassificationResult = BaseModel
    BatchClassificationResult = BaseModel
    RetryStats = BaseModel

# === RETRY & TIMEOUT DECORATOR ===

class ClassificationTimeout(Exception):
    """Custom timeout exception für Klassifikation.
    
    Wird ausgelöst wenn eine Klassifikations-Operation das konfigurierte Timeout überschreitet.
    """
    
    pass


class ClassificationRetryError(Exception):
    """Exception für fehlgeschlagene Retries.
    
    Wird ausgelöst wenn alle Retry-Versuche einer Operation fehlgeschlagen sind.
    """
    
    pass

def with_retry_and_timeout(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    timeout_seconds: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True
):
    """
    Decorator für Retry-Logik mit exponential backoff und Timeout
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_stats = {
                'attempts': 0,
                'total_delay': 0.0,
                'error_types': [],
                'start_time': time.time()
            }
            
            last_exception = None
            
            for attempt in range(max_retries + 1):
                retry_stats['attempts'] = attempt + 1
                
                try:
                    # Timeout-Wrapper für die eigentliche Funktion
                    start_time = time.time()
                    
                    if timeout_seconds > 0:
                        # Simuliert Timeout-Check (bei echten async-Calls würde man asyncio.wait_for verwenden)
                        result = func(*args, **kwargs)
                        elapsed = time.time() - start_time
                        
                        if elapsed > timeout_seconds:
                            raise ClassificationTimeout(f"Function exceeded timeout of {timeout_seconds}s")
                    else:
                        result = func(*args, **kwargs)
                    
                    # Erfolg - füge Retry-Stats hinzu falls Pydantic-Modell
                    if hasattr(result, '__dict__') and PYDANTIC_AVAILABLE:
                        if hasattr(result, 'model_fields') or isinstance(result, BaseModel):
                            result_dict = result.dict() if hasattr(result, 'dict') else result.__dict__
                            result_dict['retry_stats'] = RetryStats(
                                attempts=retry_stats['attempts'],
                                total_delay=retry_stats['total_delay'],
                                final_success=True,
                                error_types=retry_stats['error_types']
                            ).dict()
                            
                            # Rekonstruiere Objekt mit Retry-Stats
                            if hasattr(result, '__class__'):
                                return result.__class__(**result_dict)
                    
                    return result
                    
                except (ClassificationTimeout, torch.cuda.OutOfMemoryError, ConnectionError, 
                        RuntimeError) as e:
                    last_exception = e
                    error_type = type(e).__name__
                    retry_stats['error_types'].append(error_type)
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e} ({error_type})")
                    
                    # Letzter Versuch - keine weitere Verzögerung
                    if attempt == max_retries:
                        break
                    
                    # Berechne Delay mit exponential backoff
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    
                    # Jitter hinzufügen um Thundering Herd zu vermeiden
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    retry_stats['total_delay'] += delay
                    
                    logger.info(f"Retrying in {delay:.2f}s", 
                              attempt=attempt + 1, 
                              max_retries=max_retries)
                    
                    time.sleep(delay)
                
                except Exception as e:
                    # Unerwarteter Fehler - sofort abbrechen
                    logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                    raise
            
            # Alle Versuche fehlgeschlagen
            total_time = time.time() - retry_stats['start_time']
            retry_error = ClassificationRetryError(
                f"Function {func.__name__} failed after {max_retries + 1} attempts "
                f"in {total_time:.2f}s. Last error: {last_exception}"
            )
            
            logger.error(f"All retry attempts failed for {func.__name__}: {retry_stats['attempts']} attempts in {total_time:.2f}s, errors: {retry_stats['error_types']}")
            
            raise retry_error from last_exception
        
        return wrapper
    return decorator

# === UTILITY FUNCTIONS FOR PDF EXTRACTION ===

def extract_text_from_pdf(
    pdf_path: Union[str, Path], 
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SIMPLE,
    max_chunk_size: int = 1000,
    enable_chunking: bool = True
) -> ExtractedContent:
    """
    Utility-Funktion für PDF-Textextraktion außerhalb des Klassifikators.
    
    Diese Funktion trennt die PDF-Extraktion von der Klassifikation nach dem
    Single Responsibility Principle. Sie kann von Pipeline-Komponenten oder
    anderen Stellen aufgerufen werden.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        chunking_strategy: Strategie für Text-Chunking
        max_chunk_size: Maximale Chunk-Größe
        enable_chunking: Ob Chunking aktiviert werden soll
        
    Returns:
        ExtractedContent mit Text und optional Chunks
        
    Raises:
        Exception: Bei PDF-Extraktionsfehlern
    """
    extractor = EnhancedPDFExtractor(enable_chunking=enable_chunking)
    return extractor.extract_text_from_pdf(
        pdf_path,
        chunking_strategy=chunking_strategy,
        max_chunk_size=max_chunk_size
    )

def extract_multiple_pdfs(
    pdf_directory: Union[str, Path]
) -> List[ExtractedContent]:
    """
    Utility-Funktion für Batch-PDF-Extraktion.
    
    Args:
        pdf_directory: Verzeichnis mit PDF-Dateien
        
    Returns:
        Liste von ExtractedContent-Objekten
    """
    extractor = EnhancedPDFExtractor(enable_chunking=True)
    return extractor.extract_multiple_pdfs(pdf_directory)

# === ERWEITERTE CLASSIFIER-KLASSE ===

class RealMLClassifier:
    """ML-basierter Dokumentklassifizierer mit Retry-Mechanismus und Batch-Verarbeitung.
    
    Diese Klasse konzentriert sich ausschließlich auf die Klassifikation von Texten
    nach dem Single Responsibility Principle. PDF-Extraktion wird durch externe
    Utility-Funktionen gehandhabt.
    
    Attributes:
        device: PyTorch device für Model-Inferenz
        batch_size: Batch-Größe für parallele Verarbeitung
        max_retries: Maximale Anzahl Wiederholungsversuche bei Fehlern
        timeout_seconds: Timeout für einzelne Operations
        tokenizer: Hugging Face Tokenizer
        model: Trainiertes Klassifikationsmodell
        
    Note:
        PDF-Extraktion erfolgt über externe Utility-Funktionen:
        - extract_text_from_pdf(): Einzelne PDF-Extraktion
        - extract_multiple_pdfs(): Batch-PDF-Extraktion
    """
    
    def __init__(
        self,
        model_path: str = ML_MODEL_PATH,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        lazy: Optional[bool] = None,
        **kwargs
    ) -> None:
        """Initialisiert den ML-Classifier.
        
        Args:
            model_path: Pfad zum trainierten ML-Modell (Legacy-Parameter)
            model_dir: Verzeichnis mit lokalem Modell (Priority 1)
            model_name: HuggingFace Model Name (Fallback)
            batch_size: Batch-Größe für parallele Verarbeitung
            max_retries: Maximale Anzahl Wiederholungsversuche
            timeout_seconds: Timeout in Sekunden für Operations
        """
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        
        # Konfigurierbare Model-Pfade
        self.model_dir = model_dir or os.getenv("BUPROC_MODEL_DIR", "artifacts/model-v1")
        self.model_name = model_name or os.getenv("BUPROC_MODEL_NAME", "deepset/gbert-base")
        self.labels = None
        
        # Lazy loading Steuerung: env override (BUPROC_LAZY_MODELS / BU_LAZY_MODELS)
        if lazy is None:
            lazy_env = os.getenv("BUPROC_LAZY_MODELS") or os.getenv("BU_LAZY_MODELS")
            lazy = (lazy_env or "").strip().lower() in {"1", "true", "yes"}
        self._lazy = bool(lazy)

        # Defer heavy model init if lazy enabled
        if not self._lazy:
            self._initialize_model(model_path, self.model_dir, self.model_name)
        else:
            self.model = None  # type: ignore
            self.tokenizer = None  # type: ignore
            logger.info("Classifier initialized in lazy mode - model load deferred",
                        model_dir=self.model_dir, model_name=self.model_name)
        
        # PDF-Extraktion wird nun außerhalb des Klassifikators gehandhabt
        # (Single Responsibility Principle)
        
        logger.info(
            f"Enhanced Classifier initialisiert - "
            f"Device: {self.device}, "
            f"Model Dir: {self.model_dir}, "
            f"Model Name: {self.model_name}, "
            f"Batch Size: {batch_size}, "
            f"Max Retries: {max_retries}, "
            f"Timeout: {timeout_seconds}s, "
            f"Labels Available: {self.labels is not None}"
        )

    @with_retry_and_timeout(max_retries=2, timeout_seconds=60.0)
    def _initialize_model(self, legacy_model_path: str, model_dir: str, model_name: str) -> None:
        """Initialisiert das ML-Model mit konfigurierbaren Pfaden und Retry-Mechanismus.
        
        Args:
            legacy_model_path: Legacy model path (für Rückwärtskompatibilität)
            model_dir: Lokales Model-Verzeichnis (Priority 1)
            model_name: HuggingFace Model Name (Fallback)
            
        Raises:
            Exception: Bei fehlgeschlagener Model-Initialisierung
        """
        model_source = None
        model_path_to_use = None
        
        try:
            # Priority 1: Lokales Model-Verzeichnis
            if os.path.isdir(model_dir):
                model_source = "local_directory"
                model_path_to_use = model_dir
                
                logger.info(f"Lade lokales Modell aus: {model_dir}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                
                # Lade Labels falls vorhanden
                labels_path = os.path.join(model_dir, "labels.txt")
                if os.path.exists(labels_path):
                    with open(labels_path, "r", encoding="utf-8") as f:
                        self.labels = [line.strip() for line in f if line.strip()]
                    logger.info(f"Labels geladen: {len(self.labels)} Kategorien")
                else:
                    logger.warning(f"Keine labels.txt gefunden in {model_dir}")
                    self.labels = None
            
            # Priority 2: Legacy model_path falls Verzeichnis
            elif os.path.isdir(legacy_model_path):
                model_source = "legacy_directory" 
                model_path_to_use = legacy_model_path
                
                logger.info(f"Lade Modell aus Legacy-Pfad: {legacy_model_path}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(legacy_model_path, use_fast=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(legacy_model_path)
                
                # Versuche Labels zu laden
                labels_path = os.path.join(legacy_model_path, "labels.txt")
                if os.path.exists(labels_path):
                    with open(labels_path, "r", encoding="utf-8") as f:
                        self.labels = [line.strip() for line in f if line.strip()]
                    logger.info(f"Labels aus Legacy-Pfad geladen: {len(self.labels)} Kategorien")
                else:
                    self.labels = None
            
            # Priority 3: HuggingFace Hub Model (Fallback)
            else:
                model_source = "huggingface_hub"
                model_path_to_use = model_name
                
                logger.info(f"Lade HuggingFace-Modell: {model_name} (Fallback: zero-/few-shot)")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Keine lokalen Labels für HF-Modelle
                self.labels = None
                logger.info("HuggingFace-Modell geladen - verwende Modell-interne Labels")
            
            # Model auf Device verschieben und in Eval-Modus
            self.model.to(self.device)
            self.model.eval()
            
            # Erfolgreiche Initialisierung loggen
            logger.info(
                f"Model erfolgreich geladen - "
                f"Source: {model_source}, "
                f"Path: {model_path_to_use}, "
                f"Device: {self.device}, "
                f"Labels: {'custom' if self.labels else 'model-internal'}, "
                f"Label-Count: {len(self.labels) if self.labels else 'unknown'}"
            )
            
        except Exception as e:
            logger.error(
                f"Model-Initialisierung fehlgeschlagen - "
                f"Source: {model_source}, "
                f"Path: {model_path_to_use}, "
                f"Error: {e}", 
                exc_info=True
            )
            raise

    # Backwards compatibility helper for lazy loading calls
    def _load_model_and_tokenizer(self):  # pragma: no cover (simple delegation)
        if getattr(self, 'model', None) is None or getattr(self, 'tokenizer', None) is None:
            self._initialize_model(ML_MODEL_PATH, self.model_dir, self.model_name)
        return
    
    def get_category_label(self, category_id: int) -> str:
        """Gibt das menschenlesbare Label für eine Kategorie-ID zurück.
        
        Args:
            category_id: Numerische Kategorie-ID
            
        Returns:
            Menschenlesbares Label oder "Category {id}" falls nicht verfügbar
        """
        if self.labels and 0 <= category_id < len(self.labels):
            return self.labels[category_id]
        else:
            return f"Category {category_id}"
    
    def get_available_labels(self) -> Optional[List[str]]:
        """Gibt alle verfügbaren Labels zurück.
        
        Returns:
            Liste aller Labels oder None falls nicht verfügbar
        """
        return self.labels.copy() if self.labels else None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Gibt detaillierte Model-Informationen zurück.
        
        Returns:
            Dict mit Model-Details, Pfaden und Labels
        """
        return {
            "model_dir": self.model_dir,
            "model_name": self.model_name,
            "device": str(self.device),
            "labels_available": self.labels is not None,
            "label_count": len(self.labels) if self.labels else None,
            "labels": self.get_available_labels(),
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds
        }

    @with_retry_and_timeout(max_retries=3, timeout_seconds=30.0)
    def classify_text(self, text: str) -> Union[Dict[str, Any], ClassificationResult]:
        """Klassifiziert den Input-Text in vordefinierte Kategorien.
        
        Args:
            text: Rohtext, der klassifiziert werden soll
            
        Returns:
            Ein Dict mit Kategorie-Namen und Wahrscheinlichkeiten oder 
            ClassificationResult Pydantic-Model falls verfügbar
            
        Raises:
            ClassificationRetryError: Nach fehlgeschlagenen Retry-Versuchen
        """
        start_time = time.time()

        # Ensure model loaded (lazy mode)
        self._load_model_and_tokenizer()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_SEQUENCE_LENGTH
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        confidence, prediction = torch.max(probs, dim=-1)
        confidence_value = confidence.item()
        predicted_category = prediction.item()
        is_confident = confidence_value >= CONFIDENCE_THRESHOLD

        processing_time = time.time() - start_time

        result_data = {
            "category": predicted_category,
            "category_label": self.get_category_label(predicted_category),
            "confidence": confidence_value,
            "is_confident": is_confident,
            "input_type": "text",
            "text_length": len(text),
            "processing_time": processing_time,
            "model_version": "v1.0"
        }

        if PYDANTIC_AVAILABLE:
            return ClassificationResult(**result_data)
        return result_data

    def classify_batch(
        self, 
        texts: List[str], 
        batch_id: Optional[str] = None
    ) -> Union[Dict[str, Any], BatchClassificationResult]:
        """Batch-Klassifikation für bessere Performance bei vielen Texten.
        
        Args:
            texts: Liste von Texten zur Klassifikation
            batch_id: Optionale eindeutige Batch-ID für Logging
            
        Returns:
            BatchClassificationResult mit aggregierten Ergebnissen
            
        Raises:
            ValueError: Falls keine Texte übergeben werden
        """
        if not texts:
            raise ValueError("Keine Texte für Batch-Klassifikation übergeben")
        
        start_time = time.time()
        batch_id = batch_id or f"batch_{int(time.time())}"
        
        logger.info(f"Batch-Klassifikation gestartet - Batch Size: {len(texts)}, Batch ID: {batch_id}")
        
        results = []
        successful = 0
        failed = 0
        
        # Verarbeite in Batches der konfigurierten Größe
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self._process_text_batch(batch_texts, i)
            
            for result in batch_results:
                if isinstance(result, dict) and "error" in result:
                    failed += 1
                else:
                    successful += 1
                results.append(result)
        
        total_time = time.time() - start_time
        
        logger.info(
            f"Batch-Klassifikation abgeschlossen - "
            f"Total: {len(texts)}, "
            f"Successful: {successful}, "
            f"Failed: {failed}, "
            f"Time: {total_time:.2f}s"
        )
        
        result_data = {
            "total_processed": len(texts),
            "successful": successful,
            "failed": failed,
            "batch_time": total_time,
            "results": results,
            "batch_id": batch_id
        }
        
        if PYDANTIC_AVAILABLE:
            return BatchClassificationResult(**result_data)
        else:
            return result_data

    @with_retry_and_timeout(max_retries=2, timeout_seconds=45.0)
    def _process_text_batch(self, texts: List[str], start_index: int = 0) -> List[Union[Dict, ClassificationResult]]:
        """Verarbeite einen Batch von Texten gleichzeitig"""
        try:
            # Tokenisiere alle Texte in einem Batch
            # Ensure model loaded
            self._load_model_and_tokenizer()

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_SEQUENCE_LENGTH
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            confidences, predictions = torch.max(probs, dim=-1)
            
            # Erstelle Ergebnisse für jeden Text im Batch
            results = []
            for i, (text, confidence, prediction) in enumerate(zip(texts, confidences, predictions)):
                confidence_value = confidence.item()
                predicted_category = prediction.item()
                is_confident = confidence_value >= CONFIDENCE_THRESHOLD
                
                result_data = {
                    "category": predicted_category,
                    "category_label": self.get_category_label(predicted_category),
                    "confidence": confidence_value,
                    "is_confident": is_confident,
                    "input_type": "text_batch",
                    "text_length": len(text),
                    "batch_index": start_index + i,
                    "model_version": "v1.0"
                }
                
                if PYDANTIC_AVAILABLE:
                    results.append(ClassificationResult(**result_data))
                else:
                    results.append(result_data)
            
            logger.debug(f"Batch verarbeitet", batch_size=len(texts))
            return results
            
        except Exception as e:
            logger.error(f"Batch-Verarbeitung fehlgeschlagen: {e}", exc_info=True)
            # Erstelle Fehler-Ergebnisse für alle Texte im Batch
            error_results = []
            for i, text in enumerate(texts):
                error_result = {
                    "error": str(e),
                    "input_type": "text_batch",
                    "batch_index": start_index + i,
                    "text_length": len(text),
                    "category": None,
                    "confidence": 0.0,
                    "is_confident": False
                }
                error_results.append(error_result)
            
            return error_results

    @with_retry_and_timeout(max_retries=3, timeout_seconds=60.0)
    def classify_pdf(
        self, 
        pdf_path: Union[str, Path], 
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SIMPLE,
        max_chunk_size: int = 1000,
        classify_chunks_individually: bool = False
    ) -> Union[Dict, PDFClassificationResult]:
        """Klassifiziere PDF-Dokument mit optionalem Chunking und Retry-Mechanismus
        
        Diese Methode nutzt nun externe Utility-Funktionen für die PDF-Extraktion
        und konzentriert sich ausschließlich auf die Klassifikation.
        """
        try:
            # PDF-Text mit Chunking extrahieren über externe Utility-Funktion
            extracted_content = extract_text_from_pdf(
                pdf_path,
                chunking_strategy=chunking_strategy,
                max_chunk_size=max_chunk_size,
                enable_chunking=True
            )
            
            # Nutze die spezialisierte Methode für bereits extrahierten Inhalt
            return self.classify_extracted_content(extracted_content, classify_chunks_individually)
            
        except Exception as e:
            logger.error(f"PDF-Klassifikation fehlgeschlagen für {pdf_path}: {e}", exc_info=True)
            error_result = {
                "error": str(e),
                "input_type": "pdf",
                "file_path": str(pdf_path),
                "category": None,
                "confidence": 0.0,
                "is_confident": False
            }
            return error_result

    def classify_extracted_content(
        self,
        extracted_content: ExtractedContent,
        classify_chunks_individually: bool = False
    ) -> Union[Dict, PDFClassificationResult]:
        """Klassifiziere bereits extrahierten PDF-Inhalt.
        
        Diese Methode ermöglicht die Klassifikation von bereits extrahiertem Inhalt,
        ohne dass der Klassifikator selbst PDF-Extraktion durchführen muss.
        
        Args:
            extracted_content: Bereits extrahierter PDF-Inhalt
            classify_chunks_individually: Ob Chunks einzeln klassifiziert werden sollen
            
        Returns:
            Klassifikationsergebnis
        """
        logger.info("Klassifikation von bereits extrahiertem Content", 
                   file=extracted_content.file_path,
                   pages=extracted_content.page_count,
                   text_length=len(extracted_content.text),
                   chunks_created=len(extracted_content.chunks) if extracted_content.chunking_enabled else 0,
                   chunking_method=extracted_content.chunking_method)
        
        # Entscheide Klassifikationsstrategie
        if extracted_content.chunking_enabled and classify_chunks_individually:
            # Klassifiziere jeden Chunk einzeln (mit Batch-Verarbeitung)
            result = self._classify_pdf_with_chunks_batched(extracted_content)
        else:
            # Klassifiziere gesamten Text (traditionell)
            result = self._classify_pdf_traditional(extracted_content)
        
        return result

    def _classify_pdf_traditional(self, extracted_content: ExtractedContent) -> Union[Dict, PDFClassificationResult]:
        """Traditionelle PDF-Klassifikation über gesamten Text"""
        classification_result = self.classify_text(extracted_content.text)
        
        # Konvertiere zu Dict falls Pydantic-Modell
        if hasattr(classification_result, 'dict'):
            classification_data = classification_result.dict()
        else:
            classification_data = classification_result
        
        # Erweiterte Ergebnisse mit PDF-Informationen
        enhanced_data = {
            **classification_data,
            "input_type": "pdf",
            "file_path": extracted_content.file_path,
            "page_count": extracted_content.page_count,
            "extraction_method": extracted_content.extraction_method,
            "pdf_metadata": extracted_content.metadata,
            "chunking_enabled": extracted_content.chunking_enabled,
            "chunking_method": extracted_content.chunking_method
        }
        
        if PYDANTIC_AVAILABLE:
            return PDFClassificationResult(**enhanced_data)
        else:
            return enhanced_data

    def _classify_pdf_with_chunks_batched(self, extracted_content: ExtractedContent) -> Union[Dict, PDFClassificationResult]:
        """Erweiterte PDF-Klassifikation mit Chunk-basierter Batch-Analyse"""
        if not extracted_content.chunks:
            logger.warning("Keine Chunks verfügbar, fallback zu traditioneller Methode")
            return self._classify_pdf_traditional(extracted_content)
        
        logger.info("Chunk-basierte Batch-Klassifikation gestartet", chunks=len(extracted_content.chunks))
        
        # Extrahiere alle Chunk-Texte für Batch-Verarbeitung
        chunk_texts = [chunk.text for chunk in extracted_content.chunks]
        
        # Batch-Klassifikation der Chunks
        batch_result = self.classify_batch(chunk_texts, batch_id=f"pdf_chunks_{int(time.time())}")
        
        if hasattr(batch_result, 'dict'):
            batch_data = batch_result.dict()
        else:
            batch_data = batch_result
        
        # Verarbeite Batch-Ergebnisse
        chunk_results = []
        category_votes = {}
        confidence_scores = []
        
        for i, (chunk, result) in enumerate(zip(extracted_content.chunks, batch_data['results'])):
            if hasattr(result, 'dict'):
                result_data = result.dict()
            else:
                result_data = result
            
            if "error" not in result_data:
                # Erweitere Chunk-Ergebnis
                chunk_classification = {
                    "chunk_id": chunk.id,
                    "chunk_type": chunk.chunk_type,
                    "importance_score": chunk.importance_score,
                    "text_length": len(chunk.text),
                    "classification": result_data,
                    "chunk_metadata": chunk.metadata
                }
                
                chunk_results.append(chunk_classification)
                
                # Sammle Votes für finale Kategorie (gewichtet nach Importance)
                category = result_data["category"]
                confidence = result_data["confidence"]
                weight = chunk.importance_score
                
                if category not in category_votes:
                    category_votes[category] = 0
                category_votes[category] += confidence * weight
                
                confidence_scores.append(confidence)
        
        # Berechne finale Klassifikation basierend auf Chunk-Votes
        if category_votes:
            final_category = max(category_votes.items(), key=lambda x: x[1])[0]
            final_confidence = category_votes[final_category] / sum(category_votes.values())
            average_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            final_category = 0
            final_confidence = 0.0
            average_confidence = 0.0
        
        # Chunk-Analyse Statistiken
        chunk_stats = {
            "total_chunks": len(extracted_content.chunks),
            "processed_chunks": len(chunk_results),
            "failed_chunks": len(extracted_content.chunks) - len(chunk_results),
            "average_chunk_confidence": average_confidence,
            "category_distribution": category_votes,
            "high_confidence_chunks": len([r for r in chunk_results 
                                          if r["classification"]["confidence"] > CONFIDENCE_THRESHOLD]),
            "batch_processing_time": batch_data.get('batch_time', 0.0)
        }
        
        # Finale Ergebnisse
        enhanced_data = {
            "category": final_category,
            "confidence": final_confidence,
            "is_confident": final_confidence >= CONFIDENCE_THRESHOLD,
            "input_type": "pdf_chunked_batch",
            "file_path": extracted_content.file_path,
            "page_count": extracted_content.page_count,
            "extraction_method": extracted_content.extraction_method,
            "chunking_method": extracted_content.chunking_method,
            "pdf_metadata": extracted_content.metadata,
            "semantic_clusters": extracted_content.semantic_clusters,
            "chunk_analysis": chunk_stats,
            "chunk_results": chunk_results[:5],  # Nur erste 5 für Übersichtlichkeit
            "classification_strategy": "chunk_batch_voting",
            "chunking_enabled": True
        }
        
        logger.info("Chunk-basierte Batch-Klassifikation abgeschlossen",
                   final_category=final_category,
                   final_confidence=final_confidence,
                   chunks_processed=len(chunk_results),
                   batch_time=batch_data.get('batch_time', 0.0))
        
        if PYDANTIC_AVAILABLE:
            return PDFClassificationResult(**enhanced_data)
        else:
            return enhanced_data
    
    def classify_multiple_pdfs(self, pdf_directory: Union[str, Path]) -> List[Dict]:
        """Klassifiziere mehrere PDFs aus einem Verzeichnis mit Batch-Optimierung
        
        Diese Methode nutzt externe Utility-Funktionen für die PDF-Extraktion.
        """
        try:
            # Alle PDFs extrahieren über externe Utility-Funktion
            extracted_contents = extract_multiple_pdfs(pdf_directory)
            
            if not extracted_contents:
                return []
            
            # Extrahiere alle Texte für Batch-Verarbeitung
            all_texts = [content.text for content in extracted_contents]
            
            # Batch-Klassifikation aller PDF-Texte
            batch_result = self.classify_batch(all_texts, batch_id=f"pdf_batch_{int(time.time())}")
            
            if hasattr(batch_result, 'dict'):
                batch_data = batch_result.dict()
            else:
                batch_data = batch_result
            
            results = []
            
            for content, result in zip(extracted_contents, batch_data['results']):
                try:
                    if hasattr(result, 'dict'):
                        result_data = result.dict()
                    else:
                        result_data = result
                    
                    enhanced_result = {
                        **result_data,
                        "input_type": "pdf_batch",
                        "file_path": content.file_path,
                        "page_count": content.page_count,
                        "extraction_method": content.extraction_method,
                        "pdf_metadata": content.metadata
                    }
                    
                    results.append(enhanced_result)
                    
                    if "error" not in result_data:
                        logger.info("PDF-Batch-Klassifikation erfolgreich", 
                                   file=Path(content.file_path).name,
                                   category=enhanced_result["category"],
                                   confidence=enhanced_result["confidence"])
                    
                except Exception as e:
                    logger.error(f"Einzelne PDF-Klassifikation fehlgeschlagen für {content.file_path}: {e}", exc_info=True)
                    results.append({
                        "error": str(e),
                        "input_type": "pdf_batch",
                        "file_path": content.file_path,
                        "category": None,
                        "confidence": 0.0,
                        "is_confident": False
                    })
            
            logger.info("PDF-Batch-Verarbeitung abgeschlossen", 
                       total_files=len(results),
                       successful=batch_data.get('successful', 0),
                       failed=batch_data.get('failed', 0),
                       batch_time=batch_data.get('batch_time', 0.0))
            
            return results
            
        except Exception as e:
            logger.error(f"PDF-Batch-Verarbeitung fehlgeschlagen für {pdf_directory}: {e}", exc_info=True)
            return [{
                "error": str(e),
                "input_type": "pdf_batch",
                "directory": str(pdf_directory)
            }]
    
    def classify(
        self, 
        input_data: Union[str, Path, List[str]]
    ) -> Union[Dict[str, Any], ClassificationResult, BatchClassificationResult]:
        """Universelle Klassifikationsmethode - erkennt automatisch Input-Typ.
        
        Diese Methode kann Texte, PDF-Dateien, Verzeichnisse oder Listen verarbeiten
        und wählt automatisch die passende Verarbeitungsstrategie.
        
        Args:
            input_data: Text-String, Pfad zu PDF, Verzeichnis oder Liste von Texten
            
        Returns:
            Klassifikationsergebnis je nach Input-Typ
            
        Raises:
            ValueError: Bei ungültigen Input-Typen
        """
        # Liste von Texten -> Batch-Verarbeitung
        if isinstance(input_data, list):
            if all(isinstance(item, str) for item in input_data):
                return self.classify_batch(input_data)
            else:
                raise ValueError("Alle Elemente der Liste müssen Strings sein")
        
        if isinstance(input_data, (str, Path)):
            input_path = Path(input_data)
            
            # Prüfe ob es eine PDF-Datei ist
            if input_path.exists() and input_path.suffix.lower() == '.pdf':
                return self.classify_pdf(input_path)
            
            # Prüfe ob es ein Verzeichnis mit PDFs ist
            elif input_path.exists() and input_path.is_dir():
                pdf_files = list(input_path.glob("*.pdf"))
                if pdf_files:
                    return {
                        "input_type": "directory",
                        "directory_path": str(input_path),
                        "pdf_count": len(pdf_files),
                        "results": self.classify_multiple_pdfs(input_path)
                    }
                else:
                    return {
                        "error": "Keine PDF-Dateien im Verzeichnis gefunden",
                        "input_type": "directory",
                        "directory_path": str(input_path)
                    }
            
            # Ansonsten als Text behandeln
            else:
                return self.classify_text(str(input_data))
        
        # Falls direkt Text übergeben wird
        else:
            return self.classify_text(str(input_data))

    def get_health_status(self) -> Dict[str, Any]:
        """Prüft den Gesundheitsstatus des Classifiers.
        
        Führt einen Test-Klassifikationslauf durch und sammelt System-Metriken.
        
        Returns:
            Dict mit Status-Informationen, Response-Zeit und System-Details
        """
        try:
            # Test-Klassifikation für Health Check
            test_text = "Test classification for health check"
            start_time = time.time()
            test_result = self.classify_text(test_text)
            response_time = time.time() - start_time
            
            # Model-Info erweitern
            model_info = self.get_model_info()
            
            return {
                "status": "healthy",
                "model_loaded": True,
                "device": str(self.device),
                "response_time": response_time,
                "batch_size": self.batch_size,
                "max_retries": self.max_retries,
                "timeout_seconds": self.timeout_seconds,
                "pydantic_available": PYDANTIC_AVAILABLE,
                "test_classification": "passed",
                "model_info": model_info
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": False,
                "test_classification": "failed"
            }

def demo_enhanced_classifier() -> None:
    """Demo-Funktion für erweiterten Classifier mit Batch-Support und Retry-Mechanismus.
    
    Demonstriert verschiedene Features:
    - Einzelne Text-Klassifikation mit Schema-Validierung
    - Batch-Klassifikation für Performance-Vergleiche
    - PDF-Klassifikation mit Retry-Mechanismus
    - Health-Check und Performance-Metriken
    """
    print("🚀 Enhanced BU Classifier Demo - Batch, Retry & Schema Validation")
    print("================================================================")
    
    try:
        classifier = RealMLClassifier(batch_size=16, max_retries=2, timeout_seconds=20.0)
        
        # Health Check mit Model-Info
        print("\n🏥 Health Check & Model Info:")
        health = classifier.get_health_status()
        print(f"   Status: {health['status']}")
        print(f"   Device: {health.get('device', 'N/A')}")
        print(f"   Response Time: {health.get('response_time', 0):.3f}s")
        print(f"   Pydantic Available: {health.get('pydantic_available', False)}")
        
        # Model-spezifische Informationen
        model_info = health.get('model_info', {})
        print(f"   Model Dir: {model_info.get('model_dir', 'N/A')}")
        print(f"   Model Name: {model_info.get('model_name', 'N/A')}")
        print(f"   Labels Available: {model_info.get('labels_available', False)}")
        print(f"   Label Count: {model_info.get('label_count', 'N/A')}")
        
        # Zeige verfügbare Labels falls vorhanden
        available_labels = classifier.get_available_labels()
        if available_labels:
            print(f"   Available Labels: {available_labels[:3]}{'...' if len(available_labels) > 3 else ''}")
        else:
            print(f"   Using model-internal labels")
        
        # Einzelne Text-Klassifikation mit Schema
        print("\n📝 Einzelne Text-Klassifikation (mit Schema):")
        text_example = "Ich arbeite als Softwareentwickler in einer großen IT-Firma."
        result = classifier.classify_text(text_example)
        
        print(f"   Text: {text_example}")
        print(f"   Typ: {type(result).__name__}")
        
        if hasattr(result, 'dict'):
            result_data = result.dict()
            print(f"   Kategorie: {result_data['category']} ({result_data.get('category_label', 'N/A')})")
            print(f"   Confidence: {result_data['confidence']:.3f}")
            print(f"   Processing Time: {result_data.get('processing_time', 0):.3f}s")
            print(f"   Timestamp: {result_data.get('timestamp', 'N/A')}")
        else:
            print(f"   Kategorie: {result['category']} ({result.get('category_label', 'N/A')})")
            print(f"   Confidence: {result['confidence']:.3f}")
        
        # Batch-Klassifikation Demo
        print("\n📦 Batch-Klassifikation Demo:")
        batch_texts = [
            "Ich bin Softwareentwickler bei Google",
            "Als Arzt arbeite ich im Krankenhaus",
            "Meine Tätigkeit als Lehrer macht mir Spaß",
            "Marketing Manager in einem StartUp",
            "Anwalt in einer großen Kanzlei"
        ]
        
        batch_result = classifier.classify_batch(batch_texts)
        
        if hasattr(batch_result, 'dict'):
            batch_data = batch_result.dict()
            print(f"   Batch verarbeitet: {batch_data['total_processed']} Texte")
            print(f"   Erfolgreich: {batch_data['successful']}")
            print(f"   Fehlgeschlagen: {batch_data['failed']}")
            print(f"   Batch-Zeit: {batch_data['batch_time']:.3f}s")
            print(f"   Durchschnitt: {batch_data['batch_time'] / batch_data['total_processed']:.3f}s pro Text")
            
            print(f"\n   Erste 3 Ergebnisse:")
            for i, result in enumerate(batch_data['results'][:3]):
                if hasattr(result, 'dict') if PYDANTIC_AVAILABLE else isinstance(result, dict):
                    res_data = result.dict() if hasattr(result, 'dict') else result
                    label = res_data.get('category_label', f"Category {res_data['category']}")
                    print(f"     {i+1}. Kategorie: {res_data['category']} ({label}), Confidence: {res_data['confidence']:.3f}")
        
        # Universelle classify() Methode mit Liste
        print("\n🔄 Universelle Klassifikation (Liste):")
        universal_result = classifier.classify(batch_texts[:2])
        if hasattr(universal_result, 'dict'):
            print(f"   Auto-erkannt als Batch, {universal_result.dict()['total_processed']} Texte verarbeitet")
        
        # PDF-Test mit neuer entflochtener Architektur
        print("\n📄 PDF-Klassifikation mit entflochtener Architektur:")
        test_pdf = Path("tests/fixtures/sample.pdf")
        
        if test_pdf.exists():
            try:
                # Demonstration der neuen Architektur:
                # 1. Separate PDF-Extraktion über Utility-Funktion
                print(f"   🔍 Schritt 1: PDF-Extraktion (extern)")
                extracted_content = extract_text_from_pdf(test_pdf)
                print(f"   Extrahiert: {len(extracted_content.text)} Zeichen")
                
                # 2. Klassifikation bereits extrahierter Inhalte
                print(f"   🧠 Schritt 2: Klassifikation (nur ML)")
                pdf_result = classifier.classify_extracted_content(extracted_content)
                
                if hasattr(pdf_result, 'dict'):
                    pdf_data = pdf_result.dict()
                    print(f"   ✅ PDF klassifiziert (Schema: {type(pdf_result).__name__})")
                    label = pdf_data.get('category_label', f"Category {pdf_data['category']}")
                    print(f"   Kategorie: {pdf_data['category']} ({label})")
                    print(f"   Confidence: {pdf_data['confidence']:.3f}")
                    print(f"   Seiten: {pdf_data['page_count']}")
                    print(f"   🎯 Single Responsibility: Extraktion getrennt von Klassifikation")
                    
                    if 'retry_stats' in pdf_data:
                        retry_stats = pdf_data['retry_stats']
                        print(f"   Retry-Versuche: {retry_stats['attempts']}")
                        print(f"   Erfolgreich: {retry_stats['final_success']}")
                else:
                    print(f"   ✅ PDF klassifiziert (Dict)")
                    label = pdf_result.get('category_label', f"Category {pdf_result['category']}")
                    print(f"   Kategorie: {pdf_result['category']} ({label})")
                
                # 3. Alternative: Direkte PDF-Klassifikation (nutzt intern Utility)
                print(f"   🔄 Alternative: Direkte PDF-Klassifikation")
                direct_result = classifier.classify_pdf(test_pdf)
                if hasattr(direct_result, 'dict'):
                    direct_data = direct_result.dict()
                    label = direct_data.get('category_label', f"Category {direct_data['category']}")
                    print(f"   Direkter Weg auch erfolgreich: Kategorie {direct_data['category']} ({label})")
                    
            except Exception as e:
                print(f"   ❌ PDF-Klassifikation fehlgeschlagen: {e}")
        else:
            print(f"   ⚠️  Test-PDF nicht gefunden: {test_pdf}")
            print(f"   💡 Demo der entflochtenen Architektur:")
            print(f"       - extract_text_from_pdf() -> ExtractedContent")
            print(f"       - classifier.classify_extracted_content() -> Result")
            print(f"       - Trennung von Extraktion und Klassifikation")
        
        # Performance-Vergleich: Einzeln vs Batch
        print("\n⚡ Performance-Vergleich:")
        if len(batch_texts) >= 3:
            test_texts = batch_texts[:3]
            
            # Einzelverarbeitung
            start_time = time.time()
            for text in test_texts:
                classifier.classify_text(text)
            individual_time = time.time() - start_time
            
            # Batch-Verarbeitung
            start_time = time.time()
            classifier.classify_batch(test_texts)
            batch_time = time.time() - start_time
            
            speedup = (individual_time / batch_time) if batch_time > 0 else 0
            print(f"   Einzeln: {individual_time:.3f}s ({individual_time/len(test_texts):.3f}s pro Text)")
            print(f"   Batch: {batch_time:.3f}s ({batch_time/len(test_texts):.3f}s pro Text)")
            print(f"   Speedup: {speedup:.2f}x")
        
        # Teste konfigurierbare Model-Pfade
        print("\n🔧 Konfigurierbare Model-Pfade Demo:")
        print(f"   Environment Variables (Beispiel):")
        print(f"   BUPROC_MODEL_DIR={os.getenv('BUPROC_MODEL_DIR', 'artifacts/model-v1')}")
        print(f"   BUPROC_MODEL_NAME={os.getenv('BUPROC_MODEL_NAME', 'deepset/gbert-base')}")
        
        try:
            # Teste Initialisierung mit Custom-Parametern
            print(f"\n   Test: Custom Model-Parameter")
            test_classifier = RealMLClassifier(
                model_dir="custom/model/path",  # Existiert nicht, sollte fallback zu model_name
                model_name="distilbert-base-uncased",
                batch_size=8
            )
            test_info = test_classifier.get_model_info()
            print(f"   ✅ Fallback erfolgreich: {test_info['model_name']}")
            
        except Exception as e:
            print(f"   ⚠️ Custom Model Test: {e}")
        
        print("\n🎉 Enhanced Demo abgeschlossen! Batch, Retry, Schema & Konfigurierbare Pfade getestet.")
        
    except Exception as e:
        print(f"❌ Enhanced Demo fehlgeschlagen: {e}")
        print("💡 Tipps:")
        print("   - Stelle sicher, dass ein trainiertes Modell verfügbar ist")
        print("   - Oder setze BUPROC_MODEL_NAME auf ein HuggingFace-Modell")
        print("   - Beispiel: export BUPROC_MODEL_NAME=distilbert-base-uncased")

# Beispiel zur Nutzung des Enhanced Classifiers
if __name__ == "__main__":
    demo_enhanced_classifier()
