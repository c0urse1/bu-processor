"""Tests für PDF-Extraktor-Funktionalität."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

import pytest

# Import des OCR-Skip-Decorators für Tests, die echtes OCR benötigen
from .conftest import requires_tesseract

# Import der zu testenden Klassen
from bu_processor.pipeline.pdf_extractor import (
    EnhancedPDFExtractor,
    ExtractedContent,
    DocumentChunk,
    ChunkingStrategy,
    TextCleaner,
    PDFExtractionError,
    PDFCorruptedError,
    PDFTooLargeError,
    PDFPasswordProtectedError,
    PDFTextExtractionError
)


class TestEnhancedPDFExtractor:
    """Test Suite für EnhancedPDFExtractor."""
    
    @pytest.fixture
    def mock_fitz_document(self, mocker):
        """Mock für PyMuPDF fitz Document."""
        mock_doc = mocker.Mock()
        mock_doc.__len__ = mocker.Mock(return_value=3)  # 3 Seiten
        mock_doc.page_count = 3  # Add explicit page_count for range() compatibility
        mock_doc.needs_pass = False
        mock_doc.is_pdf = True
        mock_doc.metadata = {"title": "Test Document", "author": "Test Author"}
        
        # Mock pages
        mock_page1 = mocker.Mock()
        mock_page1.get_text.return_value = "Erste Seite mit Text Inhalt."
        mock_page2 = mocker.Mock()
        mock_page2.get_text.return_value = "Zweite Seite mit anderem Inhalt."
        mock_page3 = mocker.Mock()
        mock_page3.get_text.return_value = "Dritte Seite zum Abschluss."
        
        # Set up both indexing and load_page access
        mock_doc.__getitem__ = mocker.Mock(side_effect=[mock_page1, mock_page2, mock_page3])
        mock_doc.load_page.side_effect = [mock_page1, mock_page2, mock_page3]
        
        return mock_doc
    
    @pytest.fixture
    def mock_pypdf2_reader(self, mocker):
        """Mock für PyPDF2 Reader."""
        mock_reader = mocker.Mock()
        mock_reader.pages = []
        
        # Mock pages für PyPDF2
        for i in range(3):
            mock_page = mocker.Mock()
            mock_page.extract_text.return_value = f"PyPDF2 Seite {i+1} Inhalt."
            mock_reader.pages.append(mock_page)
        
        mock_reader.is_encrypted = False
        mock_reader.metadata = {"title": "PyPDF2 Test", "creator": "Test"}
        
        return mock_reader
    
    @pytest.fixture 
    def extractor(self):
        """Standard PDF-Extraktor für Tests."""
        return EnhancedPDFExtractor(enable_chunking=True, max_workers=2)
    
    def test_extractor_initialization(self):
        """Test der Extraktor-Initialisierung."""
        extractor = EnhancedPDFExtractor(
            prefer_method="pymupdf",
            enable_chunking=True,
            max_workers=4
        )
        
        assert extractor.prefer_method == "pymupdf"
        assert extractor.enable_chunking is True
        assert extractor.max_workers == 4
        assert "pymupdf" in extractor.supported_methods
        assert "pypdf2" in extractor.supported_methods
    
    def test_pdf_validation_file_not_found(self, extractor):
        """Test für nicht existierende PDF-Datei."""
        non_existent_pdf = Path("does_not_exist.pdf")
        
        with pytest.raises(FileNotFoundError):
            extractor._validate_pdf(non_existent_pdf)
    
    def test_pdf_validation_wrong_extension(self, extractor, tmp_path):
        """Test für falschen Dateityp."""
        wrong_file = tmp_path / "test.txt"
        wrong_file.write_text("Not a PDF")
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            extractor._validate_pdf(wrong_file)
    
    def test_pdf_validation_too_large(self, extractor, mocker, tmp_path):
        """Test für zu große PDF-Datei."""
        large_pdf = tmp_path / "large.pdf"
        large_pdf.write_text("fake pdf content")
        
        # Mock file size check
        mock_stat = mocker.Mock()
        mock_stat.st_size = 100 * 1024 * 1024  # 100MB (größer als Limit)
        mocker.patch.object(Path, "stat", return_value=mock_stat)
        
        with pytest.raises(PDFTooLargeError):
            extractor._validate_pdf(large_pdf)
    
    def test_extract_with_pymupdf_success(self, extractor, mocker, mock_fitz_document, sample_pdf_path):
        """Test erfolgreiche PyMuPDF-Extraktion."""
        # Mock fitz.open context manager
        mocker.patch("fitz.open", return_value=mock_fitz_document)
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "stat", return_value=mocker.Mock(st_size=1024*1024))
        
        # Mock _should_parallelize für predictable behavior
        mocker.patch.object(extractor, "_should_parallelize", return_value=False)
        
        result = extractor._extract_with_pymupdf(sample_pdf_path)
        
        assert isinstance(result, ExtractedContent)
        assert result.extraction_method == "pymupdf"
        assert result.page_count == 3
        assert result.file_path == str(sample_pdf_path)
        assert "Erste Seite" in result.text
        assert "Zweite Seite" in result.text
        assert "Dritte Seite" in result.text
    
    def test_extract_with_pypdf2_success(self, extractor, mocker, mock_pypdf2_reader, sample_pdf_path):
        """Test erfolgreiche PyPDF2-Extraktion."""
        # Mock open und PyPDF2.PdfReader
        mock_file = mocker.mock_open()
        mocker.patch("builtins.open", mock_file)
        mocker.patch("PyPDF2.PdfReader", return_value=mock_pypdf2_reader)
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "stat", return_value=mocker.Mock(st_size=1024*1024))
        
        result = extractor._extract_with_pypdf2(sample_pdf_path)
        
        assert isinstance(result, ExtractedContent)
        assert result.extraction_method == "pypdf2"
        assert result.page_count == 3
        assert "PyPDF2 Seite" in result.text
    
    def test_extract_with_password_protected_pdf(self, extractor, mocker, sample_pdf_path):
        """Test für passwort-geschützte PDF."""
        # Mock encrypted PDF
        mock_doc = mocker.Mock()
        mock_doc.needs_pass = True
        mocker.patch("fitz.open", return_value=mock_doc)
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "stat", return_value=mocker.Mock(st_size=1024*1024))
        
        with pytest.raises(PDFPasswordProtectedError):
            extractor._extract_with_pymupdf(sample_pdf_path)
    
    def test_fallback_chain_pymupdf_to_pypdf2(self, extractor, mocker, sample_pdf_path, mock_pypdf2_reader):
        """Test für Fallback von PyMuPDF zu PyPDF2."""
        # PyMuPDF schlägt fehl
        mocker.patch("fitz.open", side_effect=Exception("PyMuPDF failed"))
        
        # PyPDF2 funktioniert
        mock_file = mocker.mock_open()
        mocker.patch("builtins.open", mock_file)
        mocker.patch("PyPDF2.PdfReader", return_value=mock_pypdf2_reader)
        
        # Mocks für Validation
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "stat", return_value=mocker.Mock(st_size=1024*1024))
        mocker.patch.object(extractor, "_check_pdf_health", return_value={"page_count": 3, "has_text_content": True})
        mocker.patch.object(extractor.text_cleaner, "validate_extracted_text", return_value=True)
        
        # Test extraction
        result = extractor._extract_base_content(sample_pdf_path)
        
        assert result.extraction_method == "pypdf2"
        assert "PyPDF2 Seite" in result.text
    
    @requires_tesseract
    def test_ocr_fallback(self, extractor, mocker, sample_pdf_path):
        """Test für OCR-Fallback bei bildbasierten PDFs."""
        # Mock OCR dependencies
        mocker.patch("bu_processor.pipeline.pdf_extractor.OCR_AVAILABLE", True)
        
        # Standard extraction methods fail
        mocker.patch("fitz.open", side_effect=Exception("No text content"))
        mocker.patch("builtins.open", side_effect=Exception("PyPDF2 also failed"))
        
        # Mock OCR extraction
        mock_ocr_result = ExtractedContent(
            text="OCR extracted text from image PDF",
            page_count=2,
            file_path=str(sample_pdf_path),
            metadata={"ocr_applied": True},
            extraction_method="pymupdf_ocr"
        )
        mocker.patch.object(extractor, "_extract_with_ocr", return_value=mock_ocr_result)
        
        # Mock validation and health check
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "stat", return_value=mocker.Mock(st_size=1024*1024))
        mocker.patch.object(extractor, "_check_pdf_health", return_value={
            "page_count": 2, 
            "has_text_content": False,  # Triggers OCR
            "health_check_failed": False
        })
        
        result = extractor._extract_base_content(sample_pdf_path)
        
        assert result.extraction_method == "pymupdf_ocr"
        assert result.metadata["ocr_applied"] is True
        assert "OCR extracted" in result.text


class TestChunkingStrategies:
    """Tests für verschiedene Chunking-Strategien."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample-Text für Chunking-Tests."""
        return """
        Dies ist der erste Absatz. Er enthält wichtige Informationen über das Thema. 
        Der Text sollte in sinnvolle Chunks aufgeteilt werden.
        
        Der zweite Absatz behandelt andere Aspekte. Hier geht es um Details und Spezifikationen.
        Auch dieser Absatz sollte berücksichtigt werden.
        
        Ein dritter Absatz mit weiteren Informationen. Abschließende Bemerkungen sind hier zu finden.
        Das Ende des Beispiel-Textes.
        """.strip()
    
    def test_simple_chunking(self, sample_text):
        """Test für einfaches Chunking."""
        extractor = EnhancedPDFExtractor()
        chunks = extractor._simple_chunking(sample_text, max_chunk_size=100, overlap_size=20)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(len(chunk.text) <= 120 for chunk in chunks)  # Max size + overlap tolerance
        assert all(chunk.chunk_type == "sentence_group" for chunk in chunks)
    
    def test_simple_chunking_respects_sentence_boundaries(self, mocker):
        """Test dass Simple Chunking Satzgrenzen respektiert."""
        # Mock NLTK sentence tokenizer
        mock_sent_tokenize = mocker.patch("bu_processor.pipeline.pdf_extractor.nltk.sent_tokenize")
        mock_sent_tokenize.return_value = [
            "Erster Satz hier.",
            "Zweiter Satz mit mehr Inhalt.", 
            "Dritter und letzter Satz."
        ]
        mocker.patch("bu_processor.pipeline.pdf_extractor.NLTK_AVAILABLE", True)
        
        extractor = EnhancedPDFExtractor()
        text = "Erster Satz hier. Zweiter Satz mit mehr Inhalt. Dritter und letzter Satz."
        chunks = extractor._simple_chunking(text, max_chunk_size=50, overlap_size=10)
        
        # Verify sentences are kept together
        assert len(chunks) > 1
        for chunk in chunks:
            # Each chunk should contain complete sentences
            assert not chunk.text.startswith(" ")  # No leading spaces
            assert chunk.text.endswith(".")  # Complete sentences
    
    def test_semantic_chunking_fallback(self, sample_text):
        """Test dass Semantic Chunking auf Simple zurückfällt wenn Enhancer fehlt."""
        extractor = EnhancedPDFExtractor(enable_chunking=True)
        extractor.semantic_enhancer = None  # Simuliere fehlenden Enhancer
        
        chunks = extractor._semantic_chunking(sample_text, max_chunk_size=200)
        
        assert len(chunks) > 0
        assert all(chunk.metadata.get('semantic_ready') is True for chunk in chunks)
        assert all(chunk.chunk_type == "semantic_paragraph" for chunk in chunks)
    
    def test_hybrid_chunking_refinement(self, sample_text):
        """Test für Hybrid Chunking mit Verfeinerung."""
        extractor = EnhancedPDFExtractor(enable_chunking=True)
        chunks = extractor._apply_chunking_strategy(
            sample_text,
            ChunkingStrategy.HYBRID,
            max_chunk_size=150,
            overlap_size=30,
            page_count=1
        )
        
        assert len(chunks) > 0
        # Hybrid chunks sollten verfeinert sein
        assert all(chunk.metadata.get('semantic_refined') is True for chunk in chunks)
        assert all(chunk.chunk_type == "hybrid_chunk" for chunk in chunks)
    
    def test_chunking_strategy_none(self, sample_text):
        """Test dass NONE Strategy keine Chunks erstellt."""
        extractor = EnhancedPDFExtractor()
        chunks = extractor._apply_chunking_strategy(
            sample_text,
            ChunkingStrategy.NONE,
            max_chunk_size=100,
            overlap_size=20,
            page_count=1
        )
        
        assert len(chunks) == 0


class TestTextCleaner:
    """Tests für TextCleaner Utility-Klasse."""
    
    def test_clean_text_basic(self):
        """Test für grundlegende Text-Bereinigung."""
        dirty_text = "Text  mit   vielen   Leerzeichen\n\n\n\nund\tTabs."
        cleaned = TextCleaner.clean_text(dirty_text)
        
        assert "   " not in cleaned  # Keine mehrfachen Leerzeichen
        assert "\n\n\n" not in cleaned  # Keine mehrfachen Newlines
        assert cleaned.strip() == cleaned  # Keine führenden/nachfolgenden Spaces
    
    def test_clean_text_unicode_normalization(self):
        """Test für Unicode-Normalisierung."""
        unicode_text = "Café naïve résumé"  # Verschiedene Unicode-Zeichen
        cleaned = TextCleaner.clean_text(unicode_text)
        
        assert len(cleaned) > 0
        assert "Café" in cleaned
        assert "naïve" in cleaned
    
    def test_clean_text_remove_control_chars(self):
        """Test für Entfernung von Kontroll-Zeichen."""
        text_with_control = "Normal text\x00\x08\x7Fwith control chars"
        cleaned = TextCleaner.clean_text(text_with_control)
        
        assert "\x00" not in cleaned
        assert "\x08" not in cleaned
        assert "\x7F" not in cleaned
        assert "Normal textwith control chars" == cleaned
    
    def test_validate_extracted_text_valid(self):
        """Test für gültigen extrahierten Text."""
        valid_text = "Dies ist ein gültiger Text mit ausreichend Inhalt und guter Qualität."
        
        assert TextCleaner.validate_extracted_text(valid_text) is True
    
    def test_validate_extracted_text_too_short(self):
        """Test für zu kurzen Text."""
        short_text = "Kurz"
        
        assert TextCleaner.validate_extracted_text(short_text, min_length=20) is False
    
    def test_validate_extracted_text_unprintable_chars(self):
        """Test für Text mit zu vielen unprintable Zeichen."""
        unprintable_text = "\x00\x01\x02" * 50 + "readable text"
        
        assert TextCleaner.validate_extracted_text(unprintable_text) is False
    
    def test_detect_language_german(self):
        """Test für deutsche Sprach-Erkennung."""
        german_text = "Das ist ein deutscher Text mit den Wörtern der, die, das und ist sehr häufig."
        language = TextCleaner.detect_language(german_text)
        
        assert language == "de"
    
    def test_detect_language_english(self):
        """Test für englische Sprach-Erkennung."""
        english_text = "This is an English text with the words the, and, is very common."
        language = TextCleaner.detect_language(english_text)
        
        assert language == "en"
    
    def test_detect_language_unknown(self):
        """Test für unbekannte Sprache."""
        unknown_text = "12345 @#$%^&*"
        language = TextCleaner.detect_language(unknown_text)
        
        assert language == "unknown"


class TestPDFExtractionWithMocks:
    """Tests für PDF-Extraktion mit umfassenden Mocks."""
    
    def test_extract_text_from_pdf_with_simple_chunking(self, mocker, sample_pdf_path):
        """Test für PDF-Extraktion mit Simple Chunking."""
        # Mock validation
        extractor = EnhancedPDFExtractor(enable_chunking=True)
        mocker.patch.object(extractor, "_validate_pdf")
        mocker.patch.object(extractor, "_check_pdf_health", return_value={"page_count": 2, "has_text_content": True})
        
        # Mock base content extraction
        mock_base_content = ExtractedContent(
            text="Das ist ein langer Text der in mehrere Chunks aufgeteilt werden soll. " * 10,
            page_count=2,
            file_path=str(sample_pdf_path),
            metadata={"title": "Test"},
            extraction_method="mocked"
        )
        mocker.patch.object(extractor, "_extract_base_content", return_value=mock_base_content)
        
        result = extractor.extract_text_from_pdf(
            sample_pdf_path, 
            chunking_strategy=ChunkingStrategy.SIMPLE,
            max_chunk_size=100
        )
        
        assert result.chunking_enabled is True
        assert result.chunking_method == "simple"
        assert len(result.chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in result.chunks)
    
    def test_extract_text_from_pdf_no_chunking(self, mocker, sample_pdf_path):
        """Test für PDF-Extraktion ohne Chunking."""
        extractor = EnhancedPDFExtractor(enable_chunking=True)
        mocker.patch.object(extractor, "_validate_pdf")
        mocker.patch.object(extractor, "_check_pdf_health", return_value={"page_count": 1, "has_text_content": True})
        
        mock_base_content = ExtractedContent(
            text="Einfacher PDF-Text ohne Chunking",
            page_count=1,
            file_path=str(sample_pdf_path),
            metadata={},
            extraction_method="mocked"
        )
        mocker.patch.object(extractor, "_extract_base_content", return_value=mock_base_content)
        
        result = extractor.extract_text_from_pdf(
            sample_pdf_path,
            chunking_strategy=ChunkingStrategy.NONE
        )
        
        assert result.chunking_enabled is False
        assert len(result.chunks) == 0
        assert result.text == "Einfacher PDF-Text ohne Chunking"
    
    def test_extract_multiple_pdfs(self, mocker, tmp_path):
        """Test für Multiple-PDF-Extraktion."""
        # Erstelle temporäre PDF-Dateien
        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf3 = tmp_path / "doc3.pdf"
        
        for pdf in [pdf1, pdf2, pdf3]:
            pdf.write_text("fake pdf")
        
        extractor = EnhancedPDFExtractor()
        
        # Mock extract_text_from_pdf für jede Datei
        def mock_extract(pdf_path, **kwargs):
            return ExtractedContent(
                text=f"Extrahierter Text aus {pdf_path.name}",
                page_count=1,
                file_path=str(pdf_path),
                metadata={"processed": True},
                extraction_method="mocked"
            )
        
        mocker.patch.object(extractor, "extract_text_from_pdf", side_effect=mock_extract)
        
        results = extractor.extract_multiple_pdfs(tmp_path)
        
        assert len(results) == 3
        assert all(isinstance(content, ExtractedContent) for content in results)
        assert all("doc" in content.file_path for content in results)
    
    def test_extract_multiple_pdfs_partial_failure(self, mocker, tmp_path):
        """Test für partielle Fehler bei Multiple-PDF-Extraktion."""
        pdf1 = tmp_path / "good.pdf"
        pdf2 = tmp_path / "bad.pdf"
        
        for pdf in [pdf1, pdf2]:
            pdf.write_text("fake pdf")
        
        extractor = EnhancedPDFExtractor()
        
        def mock_extract_with_failure(pdf_path, **kwargs):
            if "bad" in str(pdf_path):
                raise PDFTextExtractionError("Extraction failed")
            return ExtractedContent(
                text=f"Text aus {pdf_path.name}",
                page_count=1,
                file_path=str(pdf_path),
                metadata={},
                extraction_method="mocked"
            )
        
        mocker.patch.object(extractor, "extract_text_from_pdf", side_effect=mock_extract_with_failure)
        
        results = extractor.extract_multiple_pdfs(tmp_path)
        
        # Nur die erfolgreiche Datei sollte zurückgegeben werden
        assert len(results) == 1
        assert "good.pdf" in results[0].file_path


class TestAdaptiveParallelization:
    """Tests für adaptive Parallelisierung."""
    
    def test_should_parallelize_small_pdf(self):
        """Test dass kleine PDFs nicht parallelisiert werden."""
        extractor = EnhancedPDFExtractor()
        
        # Mock Document mit wenigen Seiten
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=3)  # Nur 3 Seiten
        
        assert extractor._should_parallelize(mock_doc) is False
    
    def test_should_parallelize_large_pdf(self, mocker):
        """Test dass große PDFs parallelisiert werden."""
        extractor = EnhancedPDFExtractor()
        
        # Mock Document mit vielen Seiten
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=25)  # Viele Seiten
        
        # Mock page loading für Komplexitäts-Analyse
        mock_pages = []
        for i in range(3):
            mock_page = Mock()
            mock_page.get_text.return_value = "A" * 2000  # Viel Text pro Seite
            mock_pages.append(mock_page)
        
        mock_doc.load_page.side_effect = mock_pages
        
        assert extractor._should_parallelize(mock_doc) is True
    
    def test_parallel_page_extraction(self, mocker):
        """Test für parallele Seiten-Extraktion."""
        extractor = EnhancedPDFExtractor(max_workers=2)
        
        # Mock Document
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=3)
        
        # Mock pages mit verschiedenen Inhalten
        page_contents = ["Seite 1 Inhalt", "Seite 2 Inhalt", "Seite 3 Inhalt"]
        
        def mock_load_page(page_num):
            mock_page = Mock()
            mock_page.get_text.return_value = page_contents[page_num]
            return mock_page
        
        mock_doc.load_page.side_effect = mock_load_page
        
        text = extractor._extract_pages_parallel(mock_doc, 3)
        
        assert "Seite 1 Inhalt" in text
        assert "Seite 2 Inhalt" in text  
        assert "Seite 3 Inhalt" in text
        # Text sollte in korrekter Reihenfolge sein
        assert text.index("Seite 1") < text.index("Seite 2") < text.index("Seite 3")


class TestErrorHandlingAndExceptions:
    """Tests für Error Handling und Custom Exceptions."""
    
    def test_pdf_corrupted_error(self, mocker, sample_pdf_path):
        """Test für korrupte PDF-Datei."""
        import fitz
        
        mocker.patch("fitz.open", side_effect=fitz.FileDataError("Corrupted PDF"))
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "stat", return_value=mocker.Mock(st_size=1024))
        
        extractor = EnhancedPDFExtractor()
        
        with pytest.raises(PDFCorruptedError):
            extractor._extract_with_pymupdf(sample_pdf_path)
    
    def test_memory_error_handling(self, mocker, sample_pdf_path):
        """Test für Memory Error bei großen PDFs."""
        mocker.patch("fitz.open", side_effect=MemoryError("Not enough memory"))
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "stat", return_value=mocker.Mock(st_size=1024))
        
        extractor = EnhancedPDFExtractor()
        
        with pytest.raises(PDFTooLargeError):
            extractor._extract_with_pymupdf(sample_pdf_path)
    
    def test_no_extractable_text_error(self, mocker, sample_pdf_path):
        """Test wenn kein Text extrahiert werden kann."""
        # Mock Document ohne Text
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.needs_pass = False
        mock_doc.metadata = {}
        
        mock_page = Mock()
        mock_page.get_text.return_value = ""  # Kein Text
        mock_doc.load_page.return_value = mock_page
        
        mocker.patch("fitz.open", return_value=mock_doc)
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "stat", return_value=mocker.Mock(st_size=1024))
        
        extractor = EnhancedPDFExtractor()
        
        with pytest.raises(PDFTextExtractionError, match="No meaningful text extracted"):
            extractor._extract_with_pymupdf(sample_pdf_path)
    
    def test_extraction_fallback_all_methods_fail(self, mocker, sample_pdf_path):
        """Test wenn alle Extraktionsmethoden fehlschlagen."""
        extractor = EnhancedPDFExtractor()
        
        # Alle Methoden schlagen fehl
        mocker.patch("fitz.open", side_effect=Exception("PyMuPDF failed"))
        mocker.patch("builtins.open", side_effect=Exception("PyPDF2 failed"))
        mocker.patch("bu_processor.pipeline.pdf_extractor.OCR_AVAILABLE", False)  # OCR nicht verfügbar
        
        # Mock validation
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "stat", return_value=mocker.Mock(st_size=1024))
        mocker.patch.object(extractor, "_check_pdf_health", return_value={"has_text_content": False})
        
        with pytest.raises(PDFTextExtractionError, match="Kein extrahierbarer Text"):
            extractor._extract_base_content(sample_pdf_path)


class TestOCRIntegration:
    """Tests für OCR-Integration."""
    
    def test_ocr_extraction_mock(self, mocker, sample_pdf_path):
        """Test für OCR-Extraktion mit Mocks."""
        # Mock OCR dependencies
        mocker.patch("bu_processor.pipeline.pdf_extractor.OCR_AVAILABLE", True)
        
        mock_pytesseract = mocker.patch("bu_processor.pipeline.pdf_extractor.pytesseract")
        mock_pytesseract.image_to_string.return_value = "This is a very long and meaningful OCR extracted text that should definitely pass the meaningful text validation because it contains many alphanumeric characters."
        
        # Also patch the ocr object directly
        mock_ocr = mocker.patch("bu_processor.pipeline.pdf_extractor.ocr")
        mock_ocr.image_to_string.return_value = "This is a very long and meaningful OCR extracted text that should definitely pass the meaningful text validation because it contains many alphanumeric characters."
        
        # Mock PIL.Image directly instead of the imported Image
        mock_pil_image = mocker.patch("PIL.Image")
        mock_image_instance = Mock()
        mock_pil_image.open.return_value = mock_image_instance
        
        # Add Image to the module since it might not exist due to conditional import
        import bu_processor.pipeline.pdf_extractor as pdf_extractor_module
        import io
        pdf_extractor_module.Image = mock_pil_image
        pdf_extractor_module.io = io
        
        # Mock fitz document für OCR
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__enter__ = Mock(return_value=mock_doc)  # Context manager support
        mock_doc.__exit__ = Mock(return_value=None)
        
        mock_page = Mock()
        mock_pixmap = Mock()
        mock_pixmap.tobytes.return_value = b"fake_image_bytes"
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.load_page.return_value = mock_page
        
        mocker.patch("fitz.open", return_value=mock_doc)
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "stat", return_value=mocker.Mock(st_size=1024))
        
        # Mock io.BytesIO for image processing
        mock_bytesio = mocker.patch("io.BytesIO")
        mock_bytesio.return_value = Mock()
        
        extractor = EnhancedPDFExtractor()
        result = extractor._extract_with_ocr(sample_pdf_path)
        
        assert result.extraction_method == "pymupdf_ocr"
        assert result.metadata["ocr_applied"] is True
        assert "meaningful OCR extracted text" in result.text
        
        # Verify OCR calls
        mock_ocr.image_to_string.assert_called_once()
    
    def test_ocr_not_available(self, mocker, sample_pdf_path):
        """Test wenn OCR-Bibliotheken nicht verfügbar sind."""
        mocker.patch("bu_processor.pipeline.pdf_extractor.OCR_AVAILABLE", False)
        
        extractor = EnhancedPDFExtractor()
        
        with pytest.raises(PDFTextExtractionError, match="OCR-Bibliotheken.*sind nicht installiert"):
            extractor._extract_with_ocr(sample_pdf_path)


class TestSemanticEnhancement:
    """Tests für semantisches Enhancement."""
    
    def test_semantic_enhancement_with_mock_enhancer(self, mocker):
        """Test für semantisches Enhancement mit Mock."""
        # Mock semantic enhancer
        mock_enhancer = Mock()
        extractor = EnhancedPDFExtractor(enable_chunking=True)
        extractor.semantic_enhancer = mock_enhancer
        
        # Sample content mit Chunks
        chunks = [
            DocumentChunk(id="1", text="Chunk 1", start_position=0, end_position=10),
            DocumentChunk(id="2", text="Chunk 2", start_position=10, end_position=20)
        ]
        
        content = ExtractedContent(
            text="Original text",
            page_count=1,
            file_path="test.pdf",
            metadata={},
            extraction_method="test",
            chunks=chunks,
            chunking_enabled=True,
            chunking_method="simple"
        )
        
        enhanced_content = extractor._apply_semantic_enhancement(content)
        
        assert enhanced_content.semantic_clusters is not None
        assert enhanced_content.semantic_clusters["enhancement_applied"] is True
        
        # Verify chunks were enhanced
        for chunk in enhanced_content.chunks:
            assert "semantic_analysis" in chunk.metadata
            assert chunk.metadata["semantic_analysis"]["processed"] is True
    
    def test_semantic_enhancement_without_enhancer(self):
        """Test für semantisches Enhancement ohne verfügbaren Enhancer."""
        extractor = EnhancedPDFExtractor(enable_chunking=True)
        extractor.semantic_enhancer = None
        
        content = ExtractedContent(
            text="Test",
            page_count=1,
            file_path="test.pdf", 
            metadata={},
            extraction_method="test",
            chunks=[],
            chunking_enabled=True
        )
        
        result = extractor._apply_semantic_enhancement(content)
        
        # Should return content unchanged
        assert result == content


class TestPerformanceMetrics:
    """Tests für Performance-Metriken."""
    
    def test_extraction_timing(self, mocker, sample_pdf_path):
        """Test dass Extraktions-Timing gemessen wird."""
        extractor = EnhancedPDFExtractor()
        
        # Mock mit langsamer base content extraction
        def slow_extraction(pdf_path):
            import time
            time.sleep(0.1)  # Simuliere langsame Extraktion
            return ExtractedContent(
                text="Slow extracted text",
                page_count=1,
                file_path=str(pdf_path),
                metadata={},
                extraction_method="slow_mock"
            )
        
        mocker.patch.object(extractor, "_extract_base_content", side_effect=slow_extraction)
        mocker.patch.object(extractor, "_validate_pdf")
        mocker.patch.object(extractor, "_check_pdf_health", return_value={"has_text_content": True})
        
        result = extractor.extract_text_from_pdf(sample_pdf_path, chunking_strategy=ChunkingStrategy.NONE)
        
        assert result.extraction_time > 0.05  # Mindestens 50ms wegen sleep
        assert hasattr(result, 'extraction_time')


# Fixtures für gemeinsame Test-Daten
@pytest.fixture
def sample_extracted_content():
    """Sample ExtractedContent für Tests."""
    return ExtractedContent(
        text="Das ist ein Beispiel-Text für Tests mit mehreren Sätzen. " * 5,
        page_count=2,
        file_path="test.pdf",
        metadata={"title": "Test Document"},
        extraction_method="test"
    )


@pytest.fixture
def sample_document_chunks():
    """Sample DocumentChunks für Tests."""
    return [
        DocumentChunk(
            id="chunk_1",
            text="Erster Chunk mit wichtigem Inhalt.",
            start_position=0,
            end_position=35,
            chunk_type="paragraph",
            importance_score=0.8
        ),
        DocumentChunk(
            id="chunk_2", 
            text="Zweiter Chunk mit anderen Informationen.",
            start_position=35,
            end_position=75,
            chunk_type="paragraph",
            importance_score=0.6
        )
    ]


# Utility function für Test-Ausführung
def run_pdf_extractor_tests():
    """Führt alle PDF-Extraktor Tests aus."""
    return pytest.main([__file__, "-v", "--tb=short", "-k", "test_"])


if __name__ == "__main__":
    run_pdf_extractor_tests()
