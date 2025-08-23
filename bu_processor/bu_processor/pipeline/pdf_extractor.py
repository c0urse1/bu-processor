"""Erweiterte PDF-Text-Extraktion mit semantischem Chunking und OCR-Fallback."""

import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import fitz  # PyMuPDF
import PyPDF2

# Re-export ContentType for API compatibility
from .content_types import ContentType

# Import the enhanced DocumentChunk model
from ..models.chunk import DocumentChunk, create_semantic_chunk, create_paragraph_chunk

# Meaningful text validation constants
MIN_MEANINGFUL_CHARS = 10

def _is_meaningful_text(s: str) -> bool:
    """
    Check if extracted text contains meaningful content.
    
    Args:
        s: The text string to validate
        
    Returns:
        bool: True if text contains enough alphanumeric characters to be meaningful
    """
    if not s:
        return False
    # Keep alphanumerics only to avoid whitespace/punctuation-only "text"
    alpha = re.sub(r'[^A-Za-z0-9]+', '', s)
    return len(alpha) >= MIN_MEANINGFUL_CHARS

# NEU: Import für OCR-Funktionalität
try:
    import pytesseract  # noqa: F401
    from PIL import Image
    import io
    OCR_AVAILABLE = True
    ocr = pytesseract
except ImportError:
    pytesseract = None  # allows tests to patch pdf_extractor.pytesseract
    OCR_AVAILABLE = False
    import types
    ocr = types.SimpleNamespace()
    # Dummy-API, damit Tests ocr.image_to_string patchen können
    def _dummy_ocr(*args, **kwargs): 
        return ""
    ocr.image_to_string = _dummy_ocr

# NEU: Import für Satz-basiertes Chunking (nltk als optionale, robustere Alternative)
try:
    import nltk
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    import types
    import re
    nltk = types.SimpleNamespace()
    nltk.sent_tokenize = lambda text, language='german': re.split(r'(?<=[.!?])\s+', text)

# Import semantic chunking (mit fallback)
try:
    from .semantic_chunking_enhancement import SemanticClusteringEnhancer
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False

# Real semantic chunking imports
try:
    from ..semantic.embeddings import SbertEmbeddings
    from ..semantic.testing import FakeDeterministicEmbeddings
    from ..semantic.chunker import semantic_segment_sentences
    from ..semantic.tokens import approx_token_count
    from ..pipeline.chunk_entry import chunk_document_pages
    from ..pipeline.upsert_pipeline import embed_and_index_chunks, convert_chunks_to_dict
    REAL_SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    REAL_SEMANTIC_CHUNKING_AVAILABLE = False

# Import for UUID generation and timestamps
from uuid import uuid4
from datetime import datetime, timezone

# Config import mit Fallback und Logger
try:
    from ..core.config import (
        PDF_EXTRACTION_METHOD, MAX_PDF_SIZE_MB, MAX_PDF_PAGES, 
        PDF_TEXT_CLEANUP, MIN_EXTRACTED_TEXT_LENGTH, MAX_BATCH_PDF_COUNT,
        PDF_CACHE_DIR, ENABLE_PDF_CACHE, SUPPORTED_PDF_EXTENSIONS,
        PDF_FALLBACK_CHAIN, EXTRACT_PDF_METADATA, NORMALIZE_WHITESPACE,
        AUTO_DETECT_LANGUAGE, LOG_LEVEL, AppSettings, get_config
    )
    # Logger Setup aus config
    import structlog
    import logging
    
    # Configure structlog with proper log level
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
    logger = structlog.get_logger("pipeline.pdf_extractor")
    
    # Get settings for semantic chunking
    try:
        settings = AppSettings()
    except Exception:
        settings = None
    
except ImportError:
    # Fallback values und basic logging
    PDF_EXTRACTION_METHOD = "pymupdf"
    MAX_PDF_SIZE_MB = 50
    MAX_PDF_PAGES = 100
    PDF_TEXT_CLEANUP = True
    MIN_EXTRACTED_TEXT_LENGTH = 10
    MAX_BATCH_PDF_COUNT = 20
    PDF_CACHE_DIR = "cache/pdf_extractions"
    ENABLE_PDF_CACHE = True
    SUPPORTED_PDF_EXTENSIONS = [".pdf"]
    PDF_FALLBACK_CHAIN = True
    EXTRACT_PDF_METADATA = True
    NORMALIZE_WHITESPACE = True
    AUTO_DETECT_LANGUAGE = False
    LOG_LEVEL = "INFO"
    settings = None
    
    # Basic logger fallback
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("pipeline.pdf_extractor")

# Custom Exceptions for better error handling
class PDFExtractionError(Exception):
    """Base exception for PDF extraction errors"""
    pass

class PDFCorruptedError(PDFExtractionError):
    """Raised when PDF file is corrupted or unreadable"""
    pass

class PDFTooLargeError(PDFExtractionError):
    """Raised when PDF exceeds size limits"""
    pass

class PDFPasswordProtectedError(PDFExtractionError):
    """Raised when PDF is password protected"""
    pass

class PDFTextExtractionError(PDFExtractionError):
    """
    Raised when text extraction from PDF fails or produces meaningless content.
    
    Common scenarios:
    - PDF processing errors during text extraction
    - No meaningful text extracted (less than 10 alphanumeric characters)
    - Text extraction methods fail to parse content
    """
    pass

class NoExtractableTextError(PDFExtractionError):
    """Raised when no extractable text is found in PDF"""
    pass

@dataclass
class ExtractedContent:
    """Container für extrahierte PDF-Inhalte mit Chunking-Support"""
    text: str
    page_count: int
    file_path: str
    metadata: Dict
    extraction_method: str
    # Neue Felder für Chunking
    chunks: List[DocumentChunk] = field(default_factory=list)
    semantic_clusters: Optional[Dict] = None
    chunking_enabled: bool = False
    chunking_method: str = "none"
    # Performance metrics
    extraction_time: float = 0.0
    text_cleaned: bool = False
    # Page-level information for semantic chunking
    pages: List[Tuple[int, str]] = field(default_factory=list)

class ChunkingStrategy(Enum):
    """Verfügbare Chunking-Strategien.
    
    NONE: Kein Chunking, vollständiger Text
    SIMPLE: Einfache Absatz-basierte Teilung
    SEMANTIC: KI-basiertes semantisches Chunking
    HYBRID: Kombination aus einfach + semantisch
    BALANCED: Ausgewogene Strategie zwischen Performance und Qualität
    """
    
    NONE = "none"
    SIMPLE = "simple"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    BALANCED = "balanced"
class TextCleaner:
    """Utility-Klasse für Text-Bereinigung und Normalisierung.
    
    Bietet statische Methoden für Unicode-Normalisierung, Whitespace-Bereinigung
    und Text-Qualitäts-Validierung.
    """
    
    @staticmethod
    def clean_text(text: str, normalize_whitespace: bool = True) -> str:
        """Umfassende Text-Bereinigung mit Unicode-Normalisierung.
        
        Args:
            text: Zu bereinigender Text
            normalize_whitespace: Whitespace normalisieren
            
        Returns:
            Bereinigter und normalisierter Text
        """
        if not text:
            return ""
        
        try:
            # Unicode normalization (NFC form)
            cleaned = unicodedata.normalize('NFC', text)
            
            # Remove or replace problematic characters
            cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', cleaned)
            
            # Fix common PDF extraction issues
            cleaned = re.sub(r'(?<!\n)([a-z])([A-Z])', r'\1 \2', cleaned)  # Fix missing spaces
            cleaned = re.sub(r'([.!?])([A-Z])', r'\1 \2', cleaned)  # Space after punctuation
            
            if normalize_whitespace:
                # Normalize whitespace
                cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
                cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # Clean paragraph breaks
                cleaned = cleaned.strip()
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Text cleaning failed, returning original: {e}", exc_info=True)
            return text
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Simple language detection (placeholder for advanced implementation)"""
        try:
            # Basic heuristic language detection
            if not text or len(text) < 50:
                return "unknown"
            
            # Count common words/patterns for German/English
            german_indicators = ['der', 'die', 'das', 'und', 'ist', 'ein', 'eine', 'nicht', 'ich', 'auf']
            english_indicators = ['the', 'and', 'is', 'a', 'an', 'not', 'i', 'on', 'to', 'of']
            
            text_lower = text.lower()
            german_count = sum(1 for word in german_indicators if f' {word} ' in text_lower)
            english_count = sum(1 for word in english_indicators if f' {word} ' in text_lower)
            
            if german_count > english_count:
                return "de"
            elif english_count > german_count:
                return "en"
            else:
                return "unknown"
                
        except Exception:
            return "unknown"
    
    @staticmethod
    def validate_extracted_text(text: str, min_length: int = MIN_EXTRACTED_TEXT_LENGTH) -> bool:
        """Validiert ob extrahierter Text Qualitätskriterien erfüllt.
        
        Args:
            text: Zu validierender Text
            min_length: Minimale Textlänge
            
        Returns:
            True wenn Text Qualitätskriterien erfüllt
        """
        if not text or len(text.strip()) < min_length:
            return False
        
        # Check for reasonable text-to-character ratio
        printable_chars = sum(1 for c in text if c.isprintable())
        if printable_chars / len(text) < 0.7:  # At least 70% printable characters
            return False
        
        return True

class EnhancedPDFExtractor:
    """Erweiterte PDF-Text-Extraktion mit integriertem semantischem Chunking.
    
    Diese Klasse bietet robuste PDF-Text-Extraktion mit mehreren Fallback-Methoden,
    OCR-Unterstützung und erweiterten Chunking-Strategien.
    
    Attributes:
        prefer_method: Bevorzugte PDF-Extraktionsmethode
        supported_methods: Liste aller unterstützten Extraktionsmethoden
        enable_chunking: Ob Chunking-Funktionalität aktiviert ist
        max_workers: Anzahl Worker-Threads für parallele Verarbeitung
        text_cleaner: Text-Bereinigungsinstanz
        semantic_enhancer: Semantisches Enhancement (optional)
    """
    
    def __init__(
        self, 
        prefer_method: Optional[str] = None, 
        enable_chunking: bool = True, 
        max_workers: int = 4
    ) -> None:
        """Initialisiert den erweiterten PDF-Extraktor.
        
        Args:
            prefer_method: Bevorzugte Extraktionsmethode ('pymupdf' oder 'pypdf2')
            enable_chunking: Aktiviert semantisches Chunking
            max_workers: Anzahl Worker-Threads für parallele Verarbeitung
        """
        self.prefer_method = prefer_method or PDF_EXTRACTION_METHOD
        self.supported_methods = ["pymupdf", "pypdf2"]
        self.enable_chunking = enable_chunking
        self.max_workers = max_workers
        self.text_cleaner = TextCleaner()
        
        # Thread-safe lock für parallel processing
        self._lock = threading.Lock()
        
        # Semantic Enhancer initialisieren falls verfügbar
        if SEMANTIC_CHUNKING_AVAILABLE and enable_chunking:
            try:
                self.semantic_enhancer = SemanticClusteringEnhancer()
                logger.info("Semantic chunking enhancer loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load semantic enhancer: {e}", exc_info=True)
                self.semantic_enhancer = None
        else:
            self.semantic_enhancer = None
            if enable_chunking and not SEMANTIC_CHUNKING_AVAILABLE:
                logger.warning("Semantic chunking requested but not available")

        # OCR Status
        if OCR_AVAILABLE:
            logger.info("OCR engine (pytesseract) is available for image-based PDFs.")
        else:
            logger.warning("OCR engine (pytesseract) not found. Image-based PDFs cannot be processed.")
    
    def _validate_pdf(self, pdf_path: Path) -> None:
        """Validate PDF file before processing"""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF nicht gefunden: {pdf_path}")
            
        if pdf_path.suffix.lower() not in SUPPORTED_PDF_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {pdf_path.suffix}. Supported: {SUPPORTED_PDF_EXTENSIONS}")
        
        # Check file size
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_PDF_SIZE_MB:
            raise PDFTooLargeError(f"PDF too large: {file_size_mb:.1f}MB > {MAX_PDF_SIZE_MB}MB")
        
        logger.debug("PDF validation passed", file=str(pdf_path), size_mb=f"{file_size_mb:.1f}")
    
    def _check_pdf_health(self, pdf_path: Path) -> Dict:
        """Quick health check of PDF file"""
        try:
            with fitz.open(str(pdf_path)) as doc:
                page_count = len(doc)
                is_encrypted = doc.needs_pass
                has_text = False
                
                # Check first few pages for text content
                check_pages = min(3, page_count)
                for i in range(check_pages):
                    page = doc.load_page(i)
                    if page.get_text().strip():
                        has_text = True
                        break
                
                return {
                    "page_count": page_count,
                    "is_encrypted": is_encrypted,
                    "has_text_content": has_text,
                    "file_size_mb": pdf_path.stat().st_size / (1024 * 1024)
                }
                
        except Exception as e:
            logger.warning(f"PDF health check failed for {pdf_path}: {e}", exc_info=True)
            return {"health_check_failed": True, "error": str(e)}
        
    def extract_text_from_pdf(
        self, 
        pdf_path: Union[str, Path], 
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SIMPLE,
        max_chunk_size: int = 1000,
        overlap_size: int = 100
    ) -> ExtractedContent:
        """Hauptmethode für PDF-Text-Extraktion mit optionalem Chunking.
        
        Extrahiert Text aus PDF-Dateien mit konfigurierbaren Chunking-Strategien
        und robustem Error Handling.
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            chunking_strategy: Strategie für Text-Segmentierung
            max_chunk_size: Maximale Zeichen pro Chunk
            overlap_size: Überlappung zwischen Chunks in Zeichen
            
        Returns:
            ExtractedContent mit Text, Chunks und Metadaten
            
        Raises:
            PDFTooLargeError: PDF überschreitet Größenlimits
            PDFPasswordProtectedError: PDF ist passwort-geschützt
            PDFTextExtractionError: Text-Extraktion fehlgeschlagen
        """
        pdf_path = Path(pdf_path)
        start_time = time.time()
        
        # Validierung mit besseren Exceptions
        try:
            self._validate_pdf(pdf_path)
        except (FileNotFoundError, ValueError, PDFTooLargeError) as e:
            logger.error(f"PDF validation failed for {pdf_path}: {e}", exc_info=True)
            raise
        
        # PDF Health Check
        health_info = self._check_pdf_health(pdf_path)
        if health_info.get("health_check_failed"):
            logger.warning("PDF health check failed, proceeding with caution", file=str(pdf_path))
        elif health_info.get("is_encrypted"):
            raise PDFPasswordProtectedError(f"PDF is password protected: {pdf_path}")
        elif not health_info.get("has_text_content"):
            logger.warning("PDF may not contain extractable text", file=str(pdf_path))
        
        logger.info(f"PDF-Extraktion gestartet - File: {pdf_path}, Method: {self.prefer_method}, "
                   f"Chunking: {chunking_strategy.value}, Pages: {health_info.get('page_count', 'unknown')}, "
                   f"Size: {health_info.get('file_size_mb', 'unknown')} MB")
        
        # Grundlegende PDF-Extraktion mit Performance-Tracking
        try:
            base_content = self._extract_base_content(pdf_path)
            
            # Text-Bereinigung anwenden falls konfiguriert
            if PDF_TEXT_CLEANUP and base_content.text:
                original_length = len(base_content.text)
                base_content.text = self.text_cleaner.clean_text(
                    base_content.text, 
                    normalize_whitespace=NORMALIZE_WHITESPACE
                )
                base_content.text_cleaned = True
                
                logger.debug("Text cleaning applied", 
                           original_length=original_length, 
                           cleaned_length=len(base_content.text))
                
                # Validiere bereinigten Text auf Qualität und Bedeutsamkeit
                if not self.text_cleaner.validate_extracted_text(base_content.text):
                    logger.warning("Extracted text quality is poor", 
                                 text_length=len(base_content.text),
                                 file=str(pdf_path))
                
                # Check if cleaned text is still meaningful
                if not _is_meaningful_text(base_content.text):
                    raise PDFTextExtractionError("No meaningful text extracted from PDF after cleaning")
            
            # Sprach-Erkennung falls aktiviert
            if AUTO_DETECT_LANGUAGE and base_content.text:
                detected_language = self.text_cleaner.detect_language(base_content.text)
                base_content.metadata['detected_language'] = detected_language
                logger.debug("Language detected", language=detected_language)
            
            # Performance-Metriken hinzufügen
            extraction_time = time.time() - start_time
            base_content.extraction_time = extraction_time
            
        except (PDFCorruptedError, PDFPasswordProtectedError, PDFTextExtractionError) as e:
            logger.error(f"PDF extraction failed with {type(e).__name__} for {pdf_path}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during PDF extraction for {pdf_path}: {e}", exc_info=True)
            raise PDFExtractionError(f"PDF extraction failed: {e}") from e
        
        # Chunking anwenden falls gewünscht
        if chunking_strategy != ChunkingStrategy.NONE and self.enable_chunking:
            chunks = self._apply_chunking_strategy(
                base_content.text,
                chunking_strategy,
                max_chunk_size,
                overlap_size,
                base_content.page_count,
                pages=getattr(base_content, 'pages', None)
            )
            
            # Enhanced Content mit Chunks erstellen
            enhanced_content = ExtractedContent(
                text=base_content.text,
                page_count=base_content.page_count,
                file_path=base_content.file_path,
                metadata=base_content.metadata,
                extraction_method=base_content.extraction_method,
                chunks=chunks,
                chunking_enabled=True,
                chunking_method=chunking_strategy.value,
                pages=getattr(base_content, 'pages', None)
            )
            
            # Semantic Clustering falls verfügbar
            if (chunking_strategy in [ChunkingStrategy.SEMANTIC, ChunkingStrategy.HYBRID] 
                and self.semantic_enhancer):
                enhanced_content = self._apply_semantic_enhancement(enhanced_content)
            
            logger.info(f"PDF-Extraktion mit Chunking abgeschlossen - {len(chunks)} chunks created, "
                       f"Method: {chunking_strategy.value}")
            
            return enhanced_content
        else:
            # Rückgabe ohne Chunking (Kompatibilität)
            return base_content

    def extract_and_upsert_pdf(
        self,
        pdf_path: Union[str, Path],
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        title: Optional[str] = None,
        tenant: Optional[str] = None,
        embedder = None,
        index = None,
        store = None,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        B1) COMPLETE IMPLEMENTATION: Extract PDF with stable doc_id and proper metadata flow.
        
        This implements the pattern:
        1. Generate doc_id BEFORE chunking
        2. Use semantic chunking with proper page/section metadata 
        3. Upsert document with stable doc_id
        4. Return all metadata for downstream use
        
        Args:
            pdf_path: Path to PDF file
            chunking_strategy: Chunking method (SEMANTIC recommended)
            title: Document title (defaults to filename)
            tenant: Tenant identifier for multi-tenancy
            embedder: Embeddings backend
            index: Vector index
            store: Document store
            namespace: Vector namespace
            
        Returns:
            Dict with doc_id, chunk_ids, metadata, and extraction info
        """
        
        # 1) GENERATE STABLE DOC_ID FIRST
        doc_id = str(uuid4())
        pdf_path = Path(pdf_path)
        
        # B3) Generate ingestion timestamp
        ingested_at = datetime.now(timezone.utc).isoformat()
        
        # Default title to filename if not provided
        if title is None:
            title = pdf_path.stem
            
        logger.info(f"Starting B1 PDF extraction with stable doc_id: {doc_id}, ingested_at: {ingested_at}, file: {pdf_path}")
        
        try:
            # 2) EXTRACT WITH PAGE DATA
            extracted_content = self.extract_text_from_pdf(
                pdf_path,
                chunking_strategy=chunking_strategy,
                max_chunk_size=480,  # Use semantic defaults
                overlap_size=1
            )
            
            if not extracted_content.chunks:
                raise ValueError("No chunks were generated from PDF")
                
            logger.info(f"Extracted {len(extracted_content.chunks)} chunks with {extracted_content.chunking_method} chunking")
            
            # 3) ENHANCE CHUNKS WITH METADATA
            enhanced_chunks = []
            for chunk in extracted_content.chunks:
                # Ensure doc_id is set on each chunk
                chunk.doc_id = doc_id
                
                # Add source metadata including timestamps
                if not chunk.meta:
                    chunk.meta = {}
                chunk.meta.update({
                    "source_url": str(pdf_path),
                    "tenant": tenant,
                    "extraction_method": extracted_content.extraction_method,
                    "pdf_page_count": extracted_content.page_count,
                    # B3) Add ingestion timestamp to each chunk
                    "ingested_at": ingested_at
                })
                
                enhanced_chunks.append(chunk)
            
            # 4) CONVERT TO DICT FORMAT FOR UPSERT
            if REAL_SEMANTIC_CHUNKING_AVAILABLE:
                chunk_dicts = convert_chunks_to_dict(enhanced_chunks)
            else:
                # Fallback conversion
                chunk_dicts = []
                for chunk in enhanced_chunks:
                    chunk_dicts.append({
                        "text": chunk.text,
                        "page": chunk.page_start,
                        "section": chunk.section,
                        "meta": chunk.meta or {}
                    })
            
            # 5) UPSERT WITH STABLE DOC_ID (if components provided)
            result = {
                "doc_id": doc_id,
                "title": title,
                "source": str(pdf_path),
                "tenant": tenant,
                "chunks_count": len(enhanced_chunks),
                "chunking_method": extracted_content.chunking_method,
                "extraction_method": extracted_content.extraction_method,
                "page_count": extracted_content.page_count,
                "text_length": len(extracted_content.text),
                # B3) Include ingestion timestamp in result
                "ingested_at": ingested_at
            }
            
            if embedder and index and store:
                # Full upsert pipeline with timestamps
                upsert_result = embed_and_index_chunks(
                    doc_title=title,
                    doc_source=str(pdf_path),
                    doc_meta={
                        "tenant": tenant, 
                        "extraction_method": extracted_content.extraction_method,
                        # B3) Document-level timestamp will be added by upsert_pipeline
                    },
                    chunks=chunk_dicts,
                    embedder=embedder,
                    index=index,
                    store=store,
                    namespace=namespace,
                    doc_id=doc_id  # STABLE DOC_ID!
                )
                
                result.update(upsert_result)
                logger.info(f"B1 pipeline completed - doc_id: {doc_id}, chunks stored: {len(upsert_result['chunk_ids'])}")
            else:
                # Return chunks without storage (for testing/analysis)
                result["chunks"] = enhanced_chunks
                result["chunk_dicts"] = chunk_dicts
                logger.info(f"B1 extraction completed without storage - doc_id: {doc_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"B1 PDF extraction failed for {pdf_path}: {e}", exc_info=True)
            raise
    
    def _extract_base_content(self, pdf_path: Path) -> ExtractedContent:
        """Versucht Text-Extraktion mit Standard-Methoden, nutzt OCR als finalen Fallback."""
        health_info = self._check_pdf_health(pdf_path)
        
        # Standard-Extraktions-Kette
        try:
            if self.prefer_method == "pymupdf":
                content = self._extract_with_pymupdf(pdf_path)
                if self.text_cleaner.validate_extracted_text(content.text) and _is_meaningful_text(content.text):
                    return content
            elif self.prefer_method == "pypdf2":
                content = self._extract_with_pypdf2(pdf_path)
                if self.text_cleaner.validate_extracted_text(content.text) and _is_meaningful_text(content.text):
                    return content
        except Exception as e:
            logger.warning(f"Bevorzugte Methode {self.prefer_method} fehlgeschlagen: {e}")
            
        # Fallback zu anderen Methoden
        for method in self.supported_methods:
            if method != self.prefer_method:
                try:
                    logger.info(f"Fallback-Versuch - Method: {method}")
                    if method == "pymupdf":
                        content = self._extract_with_pymupdf(pdf_path)
                        if self.text_cleaner.validate_extracted_text(content.text) and _is_meaningful_text(content.text):
                            return content
                    elif method == "pypdf2":
                        content = self._extract_with_pypdf2(pdf_path)
                        if self.text_cleaner.validate_extracted_text(content.text) and _is_meaningful_text(content.text):
                            return content
                except Exception as e:
                    logger.warning(f"Fallback {method} fehlgeschlagen: {e}")
                    continue
        
        # Finaler Fallback: OCR, wenn kein Text gefunden wurde
        if not health_info.get("has_text_content") and OCR_AVAILABLE:
            logger.info(f"Kein Text im PDF gefunden. Versuche OCR-Extraktion. File: {pdf_path}")
            try:
                return self._extract_with_ocr(pdf_path)
            except Exception as ocr_error:
                logger.error("OCR-Extraktion fehlgeschlagen", error=str(ocr_error))
        
        raise PDFTextExtractionError("Kein extrahierbarer Text")
    
    def _apply_chunking_strategy(
        self, 
        text: str, 
        strategy: ChunkingStrategy, 
        max_chunk_size: int,
        overlap_size: int,
        page_count: int,
        pages: List[Tuple[int, str]] = None
    ) -> List[DocumentChunk]:
        """
        Wendet gewählte Chunking-Strategie an.
        
        Now uses unified chunking entry point for better consistency.
        """
        
        # Use unified chunking for semantic strategies
        if strategy in [ChunkingStrategy.SEMANTIC, ChunkingStrategy.HYBRID, ChunkingStrategy.BALANCED]:
            try:
                logger.debug(f"Attempting semantic chunking - REAL_SEMANTIC_CHUNKING_AVAILABLE: {REAL_SEMANTIC_CHUNKING_AVAILABLE}, pages available: {pages is not None}, pages count: {len(pages) if pages else 0}")
                
                if REAL_SEMANTIC_CHUNKING_AVAILABLE and pages:
                    # Use the new unified chunking with page data
                    logger.info(f"Using unified semantic chunking with {len(pages)} pages")
                    chunks = chunk_document_pages(
                        pages=pages,
                        doc_id=f"pdf_doc_{hash(text) % 10000}",
                        source_url=None,
                        tenant=None,
                        prefer_semantic=True
                    )
                    
                    logger.info(f"Unified semantic chunking completed - {len(chunks)} chunks created with strategy {strategy.value}")
                    return chunks
                else:
                    logger.warning(f"Pages data not available or semantic chunking not available, falling back to legacy - REAL_SEMANTIC_CHUNKING_AVAILABLE: {REAL_SEMANTIC_CHUNKING_AVAILABLE}, pages: {pages is not None}")
                
            except Exception as e:
                logger.warning(f"Unified chunking failed, falling back to legacy method: {e}")
                import traceback
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                # Fall back to legacy method for backward compatibility
                pass
        
        # Legacy chunking for simple strategy or fallback
        if strategy == ChunkingStrategy.SIMPLE:
            return self._simple_chunking(text, max_chunk_size, overlap_size)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(text, max_chunk_size, overlap_size)
        elif strategy == ChunkingStrategy.HYBRID:
            # Kombination: Erst einfach, dann semantisch verfeinern
            simple_chunks = self._simple_chunking(text, max_chunk_size, overlap_size)
            return self._refine_chunks_semantically(simple_chunks)
        else:
            # Default to simple chunking
            return self._simple_chunking(text, max_chunk_size, overlap_size)
    
    # ### VERBESSERUNG 3: SATZ-BASIERTES CHUNKING ###
    def _simple_chunking(self, text: str, max_chunk_size: int, overlap_size: int) -> List[DocumentChunk]:
        """Intelligenteres Chunking, das Satzgrenzen respektiert."""
        
        # Sätze extrahieren
        if NLTK_AVAILABLE:
            sentences = nltk.sent_tokenize(text, language='german')
        else:
            # Einfacherer Fallback mit Regex
            sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk_sentences = []
        current_length = 0
        chunk_id_counter = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_length + sentence_len + 1 > max_chunk_size and current_chunk_sentences:
                # Aktuellen Chunk abschließen
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(DocumentChunk(
                    chunk_id=f"chunk_{chunk_id_counter}",
                    text=chunk_text,
                    start_position=0,  # Positionen müssten neu berechnet werden, hier vereinfacht
                    end_position=len(chunk_text),
                    chunk_type="sentence_group"
                ))
                chunk_id_counter += 1
                
                # Overlap-Logik: Die letzten paar Sätze für den nächsten Chunk behalten
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_len + len(s) < overlap_size:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                
                current_chunk_sentences = overlap_sentences
                current_length = overlap_len

            current_chunk_sentences.append(sentence)
            current_length += sentence_len + 1

        # Den letzten verbleibenden Chunk hinzufügen
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(DocumentChunk(
                chunk_id=f"chunk_{chunk_id_counter}",
                text=chunk_text,
                start_position=0,
                end_position=len(chunk_text),
                chunk_type="sentence_group"
            ))

        logger.info(f"Sentence-based chunking completed - {len(chunks)} chunks created")
        return chunks
    
    def _semantic_chunking(self, text: str, max_chunk_size: int, overlap_size: int = 0) -> List[DocumentChunk]:
        """
        @deprecated: Use bu_processor.chunking.chunk_document() instead.
        
        This method is deprecated and will be removed in a future version.
        It now delegates to the unified chunking entry point.
        """
        import warnings
        warnings.warn(
            "_semantic_chunking is deprecated. Use bu_processor.chunking.chunk_document() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        logger.warning("Using deprecated _semantic_chunking method. Migrate to bu_processor.chunking.chunk_document()")
        
        # Delegate to new unified chunking approach
        try:
            from ..chunking import chunk_document
            
            # Get chunking parameters from settings or use defaults
            try:
                config = get_config()
                max_tokens = config.semantic.semantic_max_tokens
                sim_threshold = config.semantic.semantic_sim_threshold
                overlap_sentences = config.semantic.semantic_overlap_sentences
                model_name = config.semantic.semantic_model_name
            except Exception:
                max_tokens = max_chunk_size
                sim_threshold = 0.62
                overlap_sentences = overlap_size
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            # Use unified chunking
            chunk_texts = chunk_document(
                text=text,
                enable_semantic=None,  # Use config setting
                max_tokens=max_tokens,
                sim_threshold=sim_threshold,
                overlap_sentences=overlap_sentences,
                model_name=model_name
            )
            
            # Convert to DocumentChunk objects for backward compatibility
            chunks = []
            start_pos = 0
            for i, chunk_text in enumerate(chunk_texts):
                end_pos = start_pos + len(chunk_text)
                
                chunk = DocumentChunk(
                    chunk_id=f"deprecated_chunk_{i}",
                    text=chunk_text,
                    start_position=start_pos,
                    end_position=end_pos,
                    chunk_type="semantic",  # Keep original type for compatibility
                    importance_score=1.0,
                    meta={
                        'deprecated_method': True,
                        'migrate_to': 'bu_processor.chunking.chunk_document'
                    }
                )
                chunks.append(chunk)
                start_pos = end_pos
            
            return chunks
            
        except Exception as e:
            logger.error(f"Unified chunking failed in deprecated method: {e}")
            # Final fallback to simple chunking
            return self._simple_chunking(text, max_chunk_size, overlap_size)
    
    def _refine_chunks_semantically(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Verfeinert bestehende Chunks semantisch"""
        if not self.semantic_enhancer:
            return chunks
        
        logger.info(f"Applying semantic refinement to chunks - {len(chunks)} chunks")
        
        # Placeholder für semantische Verfeinerung
        for chunk in chunks:
            chunk.meta['semantic_refined'] = True
            chunk.chunk_type = "hybrid_chunk"
            # Erhöhe Importance Score für größere Chunks (potentiell wichtiger)
            if len(chunk.text) > 500:
                chunk.importance_score += 0.2
        
        return chunks
    
    # ### VERBESSERUNG 2: OCR-FALLBACK ###
    def _extract_with_ocr(self, pdf_path: Path) -> ExtractedContent:
        """Extrahiert Text aus einem PDF via OCR als Fallback."""
        if not OCR_AVAILABLE:
            raise PDFTextExtractionError("OCR-Bibliotheken (pytesseract, Pillow) sind nicht installiert.")
        
        try:
            text_parts = []
            pages = []
            page_count = 0
            with fitz.open(str(pdf_path)) as doc:
                page_count = len(doc)
                for page_num in range(page_count):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=300)  # Höhere DPI für bessere OCR-Ergebnisse
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    # OCR auf das Bild anwenden
                    page_text = ocr.image_to_string(img, lang='deu')  # 'deu' für Deutsch
                    text_parts.append(page_text)
                    pages.append((page_num + 1, page_text))  # 1-indexed page numbers

            full_text = "\n\n".join(text_parts).strip()
            
            # Use centralized meaningful text validation
            if not _is_meaningful_text(full_text):
                raise NoExtractableTextError("no text or not meaningful")
            
            logger.info(f"OCR Extraktion erfolgreich - Pages: {page_count}, "
                       f"Text length: {len(full_text)}, File: {pdf_path}")
            
            return ExtractedContent(
                text=full_text,
                page_count=page_count,
                file_path=str(pdf_path),
                metadata={"ocr_applied": True},
                extraction_method="pymupdf_ocr",
                pages=pages
            )
        except NoExtractableTextError as e:
            # Convert to PDFTextExtractionError at public boundary
            logger.error(f"No meaningful text extracted from PDF: {pdf_path}")
            raise PDFTextExtractionError(f"No meaningful text extracted from PDF: {pdf_path}") from e
        except Exception as e:
            raise PDFTextExtractionError(f"OCR extraction failed: {e}") from e
    
    def _apply_semantic_enhancement(self, content: ExtractedContent) -> ExtractedContent:
        """Wendet semantische Verbesserungen auf Chunks an"""
        if not self.semantic_enhancer or not content.chunks:
            return content
        
        try:
            logger.info(f"Applying semantic enhancement to chunks - {len(content.chunks)} chunks")
            
            # Hier würde die vollständige Integration mit semantic_chunking_enhancement erfolgen
            # enhanced_chunks = self.semantic_enhancer.enhance_chunks_with_semantic_clustering(...)
            
            # Für jetzt: Chunk-Metadaten erweitern
            for i, chunk in enumerate(content.chunks):
                chunk.meta['semantic_analysis'] = {
                    'processed': True,
                    'enhancement_version': '1.0',
                    'chunk_index': i,
                    'relative_importance': chunk.importance_score
                }
            
            content.semantic_clusters = {
                'total_clusters': max(1, len(content.chunks) // 3),  # Placeholder
                'enhancement_applied': True,
                'clustering_strategy': content.chunking_method
            }
            
            logger.info("Semantic enhancement completed")
            
        except Exception as e:
            logger.error(f"Semantic enhancement failed: {e}", exc_info=True)
        
        return content
    
    # ### VERBESSERUNG 1: ADAPTIVE PARALLELISIERUNG ###
    def _should_parallelize(self, doc: fitz.Document) -> bool:
        """Entscheidet adaptiv, ob Parallelisierung sinnvoll ist."""
        page_count = len(doc)
        if page_count <= 5:  # Bei sehr wenigen Seiten lohnt sich der Overhead nicht
            return False
        
        # Analysiere die Komplexität der ersten paar Seiten
        # Annahme: Komplexe Seiten (viel Text/Bilder) profitieren mehr von Parallelisierung
        try:
            sample_pages = min(page_count, 3)
            total_text_length = sum(len(doc.load_page(i).get_text()) for i in range(sample_pages))
            avg_text_per_page = total_text_length / sample_pages if sample_pages > 0 else 0
            
            # Heuristik: Parallelisieren, wenn > 20 Seiten ODER > 10 Seiten mit viel Text
            if page_count > 20 or (page_count > 10 and avg_text_per_page > 1500):
                logger.debug("Adaptive Entscheidung: Parallelisierung aktiviert.", page_count=page_count, avg_text=avg_text_per_page)
                return True
        except Exception:
            # Fallback bei Fehler in der Analyse
            return page_count > 10
        
        logger.debug("Adaptive Entscheidung: Parallelisierung deaktiviert.", page_count=page_count)
        return False
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> ExtractedContent:
        """Text-Extraktion mit PyMuPDF (empfohlen)"""
        try:
            # Some mocks don't support context manager; handle both
            doc = fitz.open(str(pdf_path))
            try:
                # Password check FIRST - before any text extraction
                if doc.needs_pass:
                    raise PDFPasswordProtectedError(f"PDF requires password: {pdf_path}")
                
                text, pages = self._extract_text_and_pages_from_fitz_doc(doc, pdf_path)
                
                # Get page count safely for mocks and real objects
                page_count = 1  # Default fallback
                if hasattr(doc, 'page_count'):
                    page_count = doc.page_count
                else:
                    try:
                        page_count = len(doc)
                    except (TypeError, AttributeError):
                        page_count = 1
                
                # Basic metadata extraction
                metadata = {}
                if EXTRACT_PDF_METADATA:
                    try:
                        metadata = dict(doc.metadata) if doc.metadata else {}
                        metadata['fitz_page_count'] = page_count
                        metadata['fitz_is_pdf'] = doc.is_pdf
                    except Exception as e:
                        logger.warning(f"Failed to extract metadata: {e}")
                
                logger.info(f"PyMuPDF Extraktion erfolgreich - Pages: {page_count}, "
                           f"Text length: {len(text)}, File: {pdf_path}")
                
                return ExtractedContent(
                    text=text.strip(),
                    page_count=page_count,
                    file_path=str(pdf_path),
                    metadata=metadata,
                    extraction_method="pymupdf",
                    pages=pages
                )
            finally:
                try:
                    doc.close()
                except Exception:
                    pass

        except fitz.FileDataError as e:
            raise PDFCorruptedError(f"PDF file is corrupted: {e}") from e
        except MemoryError:
            raise PDFTooLargeError(f"PDF too large: {pdf_path}")
        except PDFPasswordProtectedError:
            # Let password errors propagate as-is, don't convert them
            raise
        except NoExtractableTextError as e:
            # 🔴 This is what the tests expect:
            logger.error(f"No meaningful text extracted from PDF: {pdf_path}")
            raise PDFTextExtractionError(f"No meaningful text extracted from PDF: {pdf_path}") from e
        except Exception as e:
            # Generic wrap so the surface API is consistent
            raise PDFTextExtractionError(f"PyMuPDF extraction failed: {e}") from e
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}", exc_info=True)
            raise PDFTextExtractionError(f"PyMuPDF extraction failed: {e}") from e
        finally:
            try:
                if doc is not None:
                    doc.close()
            except Exception:
                pass
    
    def _extract_text_from_fitz_doc(self, doc, pdf_path: Path) -> str:
        """Extrahiert Text aus einem geöffneten fitz-Dokument"""
        text, _ = self._extract_text_and_pages_from_fitz_doc(doc, pdf_path)
        return text
    
    def _extract_text_and_pages_from_fitz_doc(self, doc, pdf_path: Path) -> Tuple[str, List[Tuple[int, str]]]:
        """Extrahiert Text und Page-Informationen aus einem geöffneten fitz-Dokument"""
        texts = []
        pages = []
        try:
            # Prefer a safe page iteration that works with real objects & mocks
            # Try different approaches to get page count
            page_count = 1  # Default fallback for mocks
            if hasattr(doc, 'page_count'):
                page_count = doc.page_count
            else:
                try:
                    page_count = len(doc)
                except (TypeError, AttributeError):
                    # If len() fails (e.g., on Mock objects), default to 1
                    page_count = 1
            
            for page_index in range(page_count):
                page = doc[page_index]
                page_text = page.get_text("text") or ""
                texts.append(page_text)
                pages.append((page_index + 1, page_text))  # 1-based page numbering
        except Exception as e:
            # Let upper layer wrap into PDFTextExtractionError
            raise NoExtractableTextError(str(e)) from e

        combined = "\n".join(texts).strip()
        if not _is_meaningful_text(combined):
            raise NoExtractableTextError("no text or not meaningful")

        return combined, pages
    
    def _extract_pages_parallel(self, doc, page_count: int) -> str:
        """Parallele Seiten-Extraktion für große PDFs"""
        def extract_page(page_num: int) -> Tuple[int, str]:
            try:
                page = doc.load_page(page_num)
                return page_num, page.get_text()
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num}: {e}")
                return page_num, ""
        
        # Thread-safe extraction
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_page = {executor.submit(extract_page, i): i for i in range(page_count)}
            page_texts = {}
            
            for future in as_completed(future_to_page):
                try:
                    page_num, page_text = future.result(timeout=30)  # 30s timeout per page
                    page_texts[page_num] = page_text
                except Exception as e:
                    page_num = future_to_page[future]
                    logger.warning(f"Page extraction failed for page {page_num}: {e}")
                    page_texts[page_num] = ""
        
        # Reassemble text in correct order
        text_parts = [page_texts.get(i, "") for i in range(page_count)]
        return "\n".join(text_parts)
    
    def _extract_pages_sequential(self, doc, page_count: int) -> str:
        """Sequentielle Seiten-Extraktion für kleinere PDFs"""
        text_parts = []
        
        for page_num in range(page_count):
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num}: {e}")
                text_parts.append("")  # Keep page order
        
        return "\n".join(text_parts)
    
    def _extract_text_and_pages_from_pypdf2(self, pdf_reader: PyPDF2.PdfReader) -> Tuple[str, List[Tuple[int, str]]]:
        """Extract combined text and page-level data from PyPDF2 reader"""
        page_count = len(pdf_reader.pages)
        text_parts = []
        pages = []
        
        for page_num in range(page_count):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                page_text = page_text if page_text else ""
                text_parts.append(page_text)
                pages.append((page_num + 1, page_text))  # 1-indexed page numbers
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num} with PyPDF2: {e}")
                text_parts.append("")
                pages.append((page_num + 1, ""))  # Keep page order
        
        combined_text = "\n".join(text_parts).strip()
        return combined_text, pages
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> ExtractedContent:
        """Text-Extraktion mit PyPDF2 (Fallback)"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                # Password/Encryption check
                if pdf_reader.is_encrypted:
                    raise PDFPasswordProtectedError(f"PDF is encrypted: {pdf_path}")
                
                # Basic metadata extraction
                metadata = {}
                if EXTRACT_PDF_METADATA:
                    try:
                        if pdf_reader.metadata:
                            # Convert PyPDF2 metadata to dict
                            for key, value in pdf_reader.metadata.items():
                                try:
                                    metadata[key] = str(value) if value else ""
                                except Exception:
                                    continue
                        metadata['pypdf2_page_count'] = page_count
                    except Exception as e:
                        logger.warning(f"Failed to extract metadata with PyPDF2: {e}")
                
                # Extract text and pages
                combined_text, pages = self._extract_text_and_pages_from_pypdf2(pdf_reader)
                
                # Check for meaningful text
                if not _is_meaningful_text(combined_text):
                    raise NoExtractableTextError("no text or not meaningful")
                
                logger.info(f"PyPDF2 Extraktion erfolgreich - Pages: {page_count}, "
                           f"Text length: {len(combined_text)}, File: {pdf_path}")
                
                return ExtractedContent(
                    text=combined_text,
                    page_count=page_count,
                    file_path=str(pdf_path),
                    metadata=metadata,
                    extraction_method="pypdf2",
                    pages=pages
                )
                
        except PyPDF2.errors.PdfReadError as e:
            raise PDFCorruptedError(f"PDF file is corrupted: {e}") from e
        except NoExtractableTextError as e:
            logger.error(f"No meaningful text extracted from PDF: {pdf_path}")
            raise PDFTextExtractionError(f"No meaningful text extracted from PDF: {pdf_path}") from e
        except Exception as e:
            raise PDFTextExtractionError(f"PyPDF2 extraction failed: {e}") from e
    
    def extract_multiple_pdfs(
        self, 
        pdf_directory: Union[str, Path],
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SIMPLE,
        max_chunk_size: int = 1000
    ) -> List[ExtractedContent]:
        """Mehrere PDFs aus einem Verzeichnis extrahieren mit Chunking.
        
        Verarbeitet alle PDF-Dateien in einem Verzeichnis parallel mit 
        konfigurierbaren Chunking-Strategien.
        
        Args:
            pdf_directory: Pfad zum Verzeichnis mit PDF-Dateien
            chunking_strategy: Chunking-Strategie für alle PDFs
            max_chunk_size: Maximale Chunk-Größe
            
        Returns:
            Liste von ExtractedContent-Objekten für erfolgreich verarbeitete PDFs
            
        Raises:
            FileNotFoundError: Verzeichnis nicht gefunden
        """
        pdf_dir = Path(pdf_directory)
        
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Verzeichnis nicht gefunden: {pdf_dir}")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("Keine PDF-Dateien gefunden", directory=str(pdf_dir))
            return []
        
        logger.info(f"Batch-Extraktion mit Chunking gestartet - {len(pdf_files)} PDFs, "
                   f"Strategy: {chunking_strategy.value}")
        
        extracted_contents = []
        
        for pdf_file in pdf_files:
            try:
                content = self.extract_text_from_pdf(
                    pdf_file, 
                    chunking_strategy=chunking_strategy,
                    max_chunk_size=max_chunk_size
                )
                extracted_contents.append(content)
                
                logger.info("PDF erfolgreich verarbeitet", 
                           file=pdf_file.name,
                           chunks_created=len(content.chunks) if content.chunking_enabled else 0)
                
            except Exception as e:
                logger.error(f"PDF-Verarbeitung fehlgeschlagen für {pdf_file.name}: {e}", exc_info=True)
                continue
        
        logger.info(f"Batch-Extraktion abgeschlossen - Successful: {len(extracted_contents)}, "
                   f"Failed: {len(pdf_files) - len(extracted_contents)}")
        
        return extracted_contents
    
    def extract_all(self, files: List[Union[str, Path]], 
                   chunking_strategy: ChunkingStrategy = ChunkingStrategy.SIMPLE) -> List[ExtractedContent]:
        """Parallelisierte Extraktion mehrerer PDF-Dateien mit ProcessPoolExecutor.
        
        Args:
            files: Liste von PDF-Dateipfaden
            chunking_strategy: Chunking-Strategie für alle PDFs
            
        Returns:
            Liste von ExtractedContent-Objekten
        """
        def extract_single_file(file_path):
            """Helper function für ProcessPool - muss top-level function sein"""
            try:
                extractor = EnhancedPDFExtractor(enable_chunking=self.enable_chunking)
                return extractor.extract_text_from_pdf(file_path, chunking_strategy=chunking_strategy)
            except Exception as e:
                logger.error(f"Extraction failed for {file_path}: {e}")
                return None
        
        if not files:
            return []
            
        logger.info("Starting parallel PDF extraction", file_count=len(files))
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(extract_single_file, files))
        
        # Filter None results (failed extractions)
        successful_results = [r for r in results if r is not None]
        
        logger.info("Parallel extraction completed", 
                   successful=len(successful_results),
                   failed=len(files) - len(successful_results))
        
        return successful_results

# Backward compatibility - alias für alten Namen
PDFExtractor = EnhancedPDFExtractor

# Standalone function für ProcessPoolExecutor (muss top-level sein)
def extract_text(file_path: Union[str, Path]) -> Optional[ExtractedContent]:
    """Standalone Funktion für ProcessPool-basierte Extraktion.
    
    Diese Funktion muss auf top-level stehen, da ProcessPoolExecutor
    die Funktion serialisieren muss.
    
    Args:
        file_path: Pfad zur PDF-Datei
        
    Returns:
        ExtractedContent oder None bei Fehler
    """
    try:
        extractor = EnhancedPDFExtractor()
        return extractor.extract_text_from_pdf(file_path)
    except Exception as e:
        logger.error(f"Extraction failed for {file_path}: {e}")
        return None

def extract_all(files: List[Union[str, Path]]) -> List[ExtractedContent]:
    """Parallelisierte Extraktion mehrerer PDF-Dateien.
    
    Args:
        files: Liste von PDF-Dateipfaden
        
    Returns:
        Liste von erfolgreich extrahierten Inhalten
    """
    if not files:
        return []
        
    logger.info("Starting ProcessPool extraction", file_count=len(files))
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(extract_text, files))
    
    # Filter erfolgreiche Extraktionen
    successful_results = [r for r in results if r is not None]
    
    logger.info("ProcessPool extraction completed", 
               successful=len(successful_results),
               failed=len(files) - len(successful_results))
    
    return successful_results

def demo_pdf_extraction_with_chunking() -> None:
    """Demo-Funktion für PDF-Extraktion mit verschiedenen Chunking-Strategien.
    
    Demonstriert:
    - Verschiedene Chunking-Strategien (NONE, SIMPLE, SEMANTIC, HYBRID)
    - Chunk-Analyse und Metadaten
    - Semantic Clustering Status
    - Performance-Vergleiche
    """
    print("🔍 Enhanced PDF-Extraktor Demo mit Chunking")
    print("=========================================")
    
    extractor = EnhancedPDFExtractor(enable_chunking=True)
    
    # Beispiel-PDF-Pfad
    test_pdf = Path("tests/fixtures/sample.pdf")
    
    if test_pdf.exists():
        try:
            print("\n📄 Teste verschiedene Chunking-Strategien:")
            
            # Test 1: Ohne Chunking
            result_none = extractor.extract_text_from_pdf(
                test_pdf, 
                chunking_strategy=ChunkingStrategy.NONE
            )
            print(f"   ✅ NONE: Text-Länge: {len(result_none.text)} Zeichen")
            
            # Test 2: Simple Chunking
            result_simple = extractor.extract_text_from_pdf(
                test_pdf, 
                chunking_strategy=ChunkingStrategy.SIMPLE,
                max_chunk_size=500,
                overlap_size=50
            )
            print(f"   ✅ SIMPLE: {len(result_simple.chunks)} Chunks erstellt")
            
            # Test 3: Semantic Chunking
            result_semantic = extractor.extract_text_from_pdf(
                test_pdf,
                chunking_strategy=ChunkingStrategy.SEMANTIC,
                max_chunk_size=800
            )
            print(f"   ✅ SEMANTIC: {len(result_semantic.chunks)} Chunks mit Semantik")
            
            # Test 4: Hybrid Chunking
            result_hybrid = extractor.extract_text_from_pdf(
                test_pdf,
                chunking_strategy=ChunkingStrategy.HYBRID,
                max_chunk_size=600
            )
            print(f"   ✅ HYBRID: {len(result_hybrid.chunks)} verfeinerte Chunks")
            
            print(f"\n📊 Chunk-Details (Simple Strategy):")
            for i, chunk in enumerate(result_simple.chunks[:3]):  # Erste 3 Chunks
                print(f"   Chunk {i+1}: {len(chunk.text)} Zeichen, Score: {chunk.importance_score:.2f}")
                print(f"   Preview: {chunk.text[:100]}...")
                print()
            
            print(f"🎯 Semantic Clustering Status:")
            if result_semantic.semantic_clusters:
                clusters = result_semantic.semantic_clusters
                print(f"   Clusters: {clusters.get('total_clusters', 'N/A')}")
                print(f"   Enhanced: {clusters.get('enhancement_applied', False)}")
            else:
                print("   Keine semantischen Cluster verfügbar")
                
        except Exception as e:
            print(f"❌ Fehler: {e}")
    else:
        print(f"⚠️  Test-PDF nicht gefunden: {test_pdf}")
        print("   Lege eine sample.pdf in tests/fixtures/ ab zum Testen")
        print("   Oder verwende: python scripts/generate_test_pdfs.py")

if __name__ == "__main__":
    demo_pdf_extraction_with_chunking()
