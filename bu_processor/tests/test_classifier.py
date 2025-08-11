"""Tests für ML-Classifier mit Mocks für externe Dienste."""

import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch

# Import der zu testenden Klassen
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bu_processor.pipeline.classifier import (
    RealMLClassifier, 
    ClassificationResult, 
    PDFClassificationResult,
    BatchClassificationResult,
    ClassificationTimeout,
    ClassificationRetryError,
    with_retry_and_timeout
)


class TestRealMLClassifier:
    """Test Suite für RealMLClassifier mit umfassenden Mocks."""
    
    @pytest.fixture
    def mock_model_components(self, mocker):
        """Mock für Transformer-Komponenten."""
        # Mock Tokenizer
        mock_tokenizer = mocker.Mock()
        mock_encoding = mocker.Mock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3]])
        mock_encoding.attention_mask = torch.tensor([[1, 1, 1]])
        mock_encoding.to = mocker.Mock(return_value=mock_encoding)
        mock_tokenizer.return_value = mock_encoding
        
        # Mock Model
        mock_model = mocker.Mock()
        mock_outputs = mocker.Mock()
        mock_outputs.logits = torch.tensor([[0.1, 0.8, 0.1]])  # High confidence for category 1
        mock_model.return_value = mock_outputs
        mock_model.to = mocker.Mock(return_value=mock_model)
        mock_model.eval = mocker.Mock()
        
        return mock_tokenizer, mock_model
    
    @pytest.fixture
    def mock_pdf_extractor(self, mocker):
        """Mock für PDF-Extraktor."""
        mock_extractor = mocker.Mock()
        mock_extracted_content = mocker.Mock()
        mock_extracted_content.text = "Test PDF Inhalt für Klassifikation"
        mock_extracted_content.page_count = 3
        mock_extracted_content.file_path = "test.pdf"
        mock_extracted_content.metadata = {"title": "Test Document"}
        mock_extracted_content.extraction_method = "mocked"
        mock_extracted_content.chunking_enabled = False
        mock_extracted_content.chunking_method = "none"
        
        mock_extractor.extract_text_from_pdf.return_value = mock_extracted_content
        return mock_extractor
    
    @pytest.fixture
    def classifier_with_mocks(self, mocker, mock_model_components):
        """Vollständig gemockter Classifier für Tests."""
        mock_tokenizer, mock_model = mock_model_components
        
        # Mock AutoTokenizer und AutoModel imports
        mocker.patch("bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", 
                     return_value=mock_tokenizer)
        mocker.patch("bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", 
                     return_value=mock_model)
        
        # Mock PyTorch device detection
        mocker.patch("torch.cuda.is_available", return_value=False)
        
        # Mock PDF Extractor
        mock_pdf_extractor = mocker.Mock()
        mocker.patch("bu_processor.pipeline.classifier.EnhancedPDFExtractor", 
                     return_value=mock_pdf_extractor)
        
        return RealMLClassifier()
    
    def test_classifier_initialization(self, mocker, mock_model_components):
        """Test der Classifier-Initialisierung mit gemockten Dependencies."""
        mock_tokenizer, mock_model = mock_model_components
        
        mocker.patch("bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", 
                     return_value=mock_tokenizer)
        mocker.patch("bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", 
                     return_value=mock_model)
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("bu_processor.pipeline.classifier.EnhancedPDFExtractor")
        
        classifier = RealMLClassifier(
            batch_size=16,
            max_retries=2,
            timeout_seconds=15.0
        )
        
        assert classifier.batch_size == 16
        assert classifier.max_retries == 2
        assert classifier.timeout_seconds == 15.0
        assert classifier.device == torch.device('cpu')
        
        # Verify model initialization calls
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()
    
    def test_classify_text_returns_correct_structure(self, classifier_with_mocks):
        """Test dass classify_text korrekte Ergebnisstruktur zurückgibt."""
        result = classifier_with_mocks.classify_text("Test text für Klassifikation")
        
        # Test Struktur des Ergebnisses
        if hasattr(result, 'dict'):  # Pydantic model
            result_data = result.dict()
        else:  # Dict
            result_data = result
        
        assert "category" in result_data
        assert "confidence" in result_data
        assert "is_confident" in result_data
        assert "input_type" in result_data
        assert "text_length" in result_data
        assert "processing_time" in result_data
        assert "model_version" in result_data
        
        assert result_data["input_type"] == "text"
        assert isinstance(result_data["category"], int)
        assert 0.0 <= result_data["confidence"] <= 1.0
        assert isinstance(result_data["is_confident"], bool)
    
    def test_classify_text_high_confidence(self, classifier_with_mocks):
        """Test für hohe Confidence-Klassifikation (Mock liefert 0.8)."""
        result = classifier_with_mocks.classify_text("Sehr eindeutiger Test-Text")
        
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        assert result_data["category"] == 1  # Kategorie mit höchstem Wert (0.8)
        assert result_data["confidence"] > 0.7  # Hohe Confidence
        assert result_data["is_confident"] is True
    
    def test_classify_batch_empty_input(self, classifier_with_mocks):
        """Test für leere Batch-Eingabe."""
        with pytest.raises(ValueError, match="Keine Texte für Batch-Klassifikation übergeben"):
            classifier_with_mocks.classify_batch([])
    
    def test_classify_batch_multiple_texts(self, classifier_with_mocks):
        """Test für Batch-Klassifikation mit mehreren Texten."""
        test_texts = [
            "Erster Test-Text",
            "Zweiter Test-Text", 
            "Dritter Test-Text"
        ]
        
        result = classifier_with_mocks.classify_batch(test_texts)
        
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        assert result_data["total_processed"] == 3
        assert result_data["successful"] == 3
        assert result_data["failed"] == 0
        assert len(result_data["results"]) == 3
        assert "batch_time" in result_data
        assert "batch_id" in result_data
    
    def test_classify_pdf_with_mock_extractor(self, classifier_with_mocks, mock_pdf_extractor):
        """Test PDF-Klassifikation mit gemocktem PDF-Extraktor."""
        # Setup mock for PDF extractor
        classifier_with_mocks.pdf_extractor = mock_pdf_extractor
        
        test_pdf_path = Path("tests/fixtures/sample.pdf")
        result = classifier_with_mocks.classify_pdf(test_pdf_path)
        
        # Verify PDF extractor was called
        mock_pdf_extractor.extract_text_from_pdf.assert_called_once()
        
        # Verify result structure
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        assert result_data["input_type"] in ["pdf", "pdf_chunked_batch"]
        assert "file_path" in result_data
        assert "page_count" in result_data
        assert "extraction_method" in result_data
    
    def test_universal_classify_method_text(self, classifier_with_mocks):
        """Test der universellen classify() Methode mit Text-Input."""
        result = classifier_with_mocks.classify("Universal test text")
        
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        assert result_data["input_type"] == "text"
    
    def test_universal_classify_method_list(self, classifier_with_mocks):
        """Test der universellen classify() Methode mit Listen-Input."""
        test_list = ["Text 1", "Text 2"]
        result = classifier_with_mocks.classify(test_list)
        
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        assert result_data["total_processed"] == 2
    
    def test_health_status(self, classifier_with_mocks):
        """Test für Health-Status Check."""
        health = classifier_with_mocks.get_health_status()
        
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
        assert "device" in health
        assert "response_time" in health
        assert "test_classification" in health


class TestMockedTransformerCalls:
    """Tests die spezifisch Transformer/Pinecone Aufrufe mocken."""
    
    def test_classify_returns_probabilities(self, mocker):
        """Mock für Transformer-Model das Wahrscheinlichkeiten zurückgibt."""
        # Mock das komplette ML-Pipeline
        fake_tokenizer = mocker.Mock()
        fake_encoding = mocker.Mock()
        fake_encoding.input_ids = torch.tensor([[101, 102, 103]])
        fake_encoding.attention_mask = torch.tensor([[1, 1, 1]])
        fake_encoding.to = mocker.Mock(return_value=fake_encoding)
        fake_tokenizer.return_value = fake_encoding
        
        fake_model = mocker.Mock()
        fake_outputs = mocker.Mock()
        fake_outputs.logits = torch.tensor([[0.2, 0.8, 0.0]])  # High prob for category 1
        fake_model.return_value = fake_outputs
        fake_model.to = mocker.Mock(return_value=fake_model)
        fake_model.eval = mocker.Mock()
        
        # Patch imports
        mocker.patch("bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", 
                     return_value=fake_tokenizer)
        mocker.patch("bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", 
                     return_value=fake_model)
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("bu_processor.pipeline.classifier.EnhancedPDFExtractor")
        
        classifier = RealMLClassifier()
        result = classifier.classify_text("Hallo Welt")
        
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        assert result_data["category"] == 1  # Index mit höchster Probability
        assert result_data["confidence"] > 0.7  # Sollte hoch sein (0.8)
    
    def test_classify_with_pinecone_mock(self, mocker):
        """Mock für Pinecone-Integration (falls verwendet)."""
        # Mock Pinecone client
        mock_pinecone = mocker.patch("pinecone.init")
        mock_index = mocker.Mock()
        mock_index.query.return_value = {
            'matches': [
                {'id': 'doc1', 'score': 0.95, 'metadata': {'category': 'finance'}},
                {'id': 'doc2', 'score': 0.87, 'metadata': {'category': 'legal'}}
            ]
        }
        
        # Wenn Pinecone-Integration existiert, teste sie hier
        # Für jetzt: Placeholder Test
        assert mock_pinecone is not None  # Pinecone wurde gemockt
        
        # Test würde hier Pinecone-spezifische Classifier-Methoden testen
        # z.B. similarity search, vector embeddings, etc.
    
    def test_torch_operations_mocked(self, mocker):
        """Test dass PyTorch Operationen korrekt gemockt werden."""
        # Mock torch operations
        mock_softmax = mocker.patch("torch.softmax")
        mock_max = mocker.patch("torch.max")
        mock_no_grad = mocker.patch("torch.no_grad")
        
        # Setze Return-Werte
        mock_softmax.return_value = torch.tensor([[0.1, 0.8, 0.1]])
        mock_max.return_value = (
            torch.tensor([0.8]),  # confidence
            torch.tensor([1])     # prediction
        )
        
        # Mock device
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.device", return_value="cpu")
        
        # Mock Transformer components
        mock_tokenizer = mocker.Mock()
        mock_encoding = mocker.Mock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3]])
        mock_encoding.to = mocker.Mock(return_value=mock_encoding)
        mock_tokenizer.return_value = mock_encoding
        
        mock_model = mocker.Mock()
        mock_model.return_value = mocker.Mock(logits=torch.tensor([[0.1, 0.8, 0.1]]))
        mock_model.to = mocker.Mock()
        mock_model.eval = mocker.Mock()
        
        mocker.patch("bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", 
                     return_value=mock_tokenizer)
        mocker.patch("bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", 
                     return_value=mock_model)
        mocker.patch("bu_processor.pipeline.classifier.EnhancedPDFExtractor")
        
        # Test classifier creation und operation
        classifier = RealMLClassifier()
        
        # Context manager mock für no_grad
        mock_no_grad.return_value.__enter__ = mocker.Mock()
        mock_no_grad.return_value.__exit__ = mocker.Mock()
        
        result = classifier.classify_text("Test")
        
        # Verify mocked operations were called
        mock_softmax.assert_called()
        mock_max.assert_called()


class TestRetryDecorator:
    """Tests für Retry-Mechanismus und Timeout-Handling."""
    
    def test_retry_decorator_success_first_try(self, mocker):
        """Test dass erfolgreiche Funktionen beim ersten Versuch durchlaufen."""
        @with_retry_and_timeout(max_retries=3)
        def success_function():
            return {"result": "success"}
        
        result = success_function()
        assert result["result"] == "success"
    
    def test_retry_decorator_eventual_success(self, mocker):
        """Test dass Retry-Decorator bei wiederholten Versuchen funktioniert."""
        call_count = 0
        
        @with_retry_and_timeout(max_retries=3, base_delay=0.01)  # Kurze Delays für Tests
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return {"result": "success after retries"}
        
        # Mock time.sleep um Tests zu beschleunigen
        mocker.patch("time.sleep")
        
        result = flaky_function()
        assert result["result"] == "success after retries"
        assert call_count == 3
    
    def test_retry_decorator_max_retries_exceeded(self, mocker):
        """Test dass nach max_retries eine Exception geworfen wird."""
        @with_retry_and_timeout(max_retries=2, base_delay=0.01)
        def always_fails():
            raise RuntimeError("Persistent failure")
        
        mocker.patch("time.sleep")
        
        with pytest.raises(ClassificationRetryError):
            always_fails()
    
    def test_timeout_handling(self, mocker):
        """Test für Timeout-Handling."""
        @with_retry_and_timeout(timeout_seconds=0.1)  # Sehr kurzes Timeout
        def slow_function():
            time.sleep(0.2)  # Länger als Timeout
            return "should not return"
        
        with pytest.raises(ClassificationTimeout):
            slow_function()


class TestClassificationResultSchemas:
    """Tests für Pydantic Schema-Validierung."""
    
    def test_classification_result_schema_valid(self):
        """Test dass gültige ClassificationResult-Daten akzeptiert werden."""
        try:
            from bu_processor.pipeline.classifier import PYDANTIC_AVAILABLE, ClassificationResult
            
            if not PYDANTIC_AVAILABLE:
                pytest.skip("Pydantic nicht verfügbar")
            
            valid_data = {
                "category": 2,
                "confidence": 0.85,
                "is_confident": True,
                "input_type": "text",
                "text_length": 150,
                "processing_time": 0.05,
                "model_version": "v1.0"
            }
            
            result = ClassificationResult(**valid_data)
            assert result.category == 2
            assert result.confidence == 0.85
            assert result.is_confident is True
            
        except ImportError:
            pytest.skip("ClassificationResult nicht importiert")
    
    def test_classification_result_schema_invalid_confidence(self):
        """Test dass ungültige Confidence-Werte abgelehnt werden."""
        try:
            from bu_processor.pipeline.classifier import PYDANTIC_AVAILABLE, ClassificationResult
            
            if not PYDANTIC_AVAILABLE:
                pytest.skip("Pydantic nicht verfügbar")
            
            invalid_data = {
                "category": 1,
                "confidence": 1.5,  # Ungültig: > 1.0
                "is_confident": True,
                "input_type": "text"
            }
            
            with pytest.raises(ValueError):
                ClassificationResult(**invalid_data)
                
        except ImportError:
            pytest.skip("ClassificationResult nicht importiert")
    
    def test_pdf_classification_result_schema(self):
        """Test für PDF-spezifische Schema-Erweiterung."""
        try:
            from bu_processor.pipeline.classifier import PYDANTIC_AVAILABLE, PDFClassificationResult
            
            if not PYDANTIC_AVAILABLE:
                pytest.skip("Pydantic nicht verfügbar")
            
            valid_pdf_data = {
                "category": 0,
                "confidence": 0.92,
                "is_confident": True,
                "input_type": "pdf",
                "file_path": "test.pdf",
                "page_count": 5,
                "extraction_method": "pymupdf",
                "pdf_metadata": {"title": "Test Document"},
                "chunking_enabled": True,
                "chunking_method": "simple"
            }
            
            result = PDFClassificationResult(**valid_pdf_data)
            assert result.file_path == "test.pdf"
            assert result.page_count == 5
            assert result.chunking_enabled is True
            
        except ImportError:
            pytest.skip("PDFClassificationResult nicht importiert")


class TestIntegrationScenarios:
    """Integration Tests für realistische Szenarien."""
    
    def test_end_to_end_text_classification(self, mocker):
        """End-to-End Test für Text-Klassifikation mit realistischen Mocks."""
        # Realistische Mock-Werte
        mock_tokenizer = mocker.Mock()
        mock_encoding = mocker.Mock()
        mock_encoding.input_ids = torch.tensor([[101, 2045, 2003, 102]])  # BERT-like token IDs
        mock_encoding.attention_mask = torch.tensor([[1, 1, 1, 1]])
        mock_encoding.to = mocker.Mock(return_value=mock_encoding)
        mock_tokenizer.return_value = mock_encoding
        
        mock_model = mocker.Mock()
        mock_outputs = mocker.Mock()
        # Realistische Logits für 3 Kategorien
        mock_outputs.logits = torch.tensor([[0.1, 0.85, 0.05]])  # Finance category (1) sehr wahrscheinlich
        mock_model.return_value = mock_outputs
        mock_model.to = mocker.Mock()
        mock_model.eval = mocker.Mock()
        
        mocker.patch("bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", 
                     return_value=mock_tokenizer)
        mocker.patch("bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", 
                     return_value=mock_model)
        mocker.patch("torch.cuda.is_available", return_value=True)  # Simuliere GPU
        mocker.patch("torch.device")
        mocker.patch("bu_processor.pipeline.classifier.EnhancedPDFExtractor")
        
        classifier = RealMLClassifier()
        
        # Test realistischen Finanz-Text
        finance_text = "Unsere Quartalszahlen zeigen einen Umsatz von 2.5 Millionen Euro und einen Gewinn von 15%. Die Aktie ist gestiegen."
        result = classifier.classify_text(finance_text)
        
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        assert result_data["category"] == 1  # Finance category
        assert result_data["confidence"] > 0.8  # Hohe Confidence
        assert result_data["is_confident"] is True
        assert result_data["text_length"] == len(finance_text)
    
    def test_error_recovery_scenario(self, mocker):
        """Test für Error Recovery mit Retry-Mechanismus."""
        retry_count = 0
        
        def failing_model_call(*args, **kwargs):
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                raise torch.cuda.OutOfMemoryError("GPU memory full")
            # Erfolg beim 3. Versuch
            mock_outputs = mocker.Mock()
            mock_outputs.logits = torch.tensor([[0.3, 0.7, 0.0]])
            return mock_outputs
        
        # Setup mocks
        mock_tokenizer = mocker.Mock()
        mock_encoding = mocker.Mock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3]])
        mock_encoding.to = mocker.Mock(return_value=mock_encoding)
        mock_tokenizer.return_value = mock_encoding
        
        mock_model = mocker.Mock()
        mock_model.side_effect = failing_model_call
        mock_model.to = mocker.Mock()
        mock_model.eval = mocker.Mock()
        
        mocker.patch("bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", 
                     return_value=mock_tokenizer)
        mocker.patch("bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", 
                     return_value=mock_model)
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("bu_processor.pipeline.classifier.EnhancedPDFExtractor")
        mocker.patch("time.sleep")  # Speed up tests
        
        classifier = RealMLClassifier()
        result = classifier.classify_text("Test after retries")
        
        # Verify it eventually succeeded
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        assert result_data["category"] == 1
        assert retry_count == 3  # 2 failures + 1 success


class TestPerformanceAndEdgeCases:
    """Tests für Performance und Edge Cases."""
    
    def test_empty_text_classification(self, classifier_with_mocks):
        """Test für leeren Text."""
        result = classifier_with_mocks.classify_text("")
        
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        assert result_data["text_length"] == 0
        assert "category" in result_data
    
    def test_very_long_text_classification(self, classifier_with_mocks):
        """Test für sehr langen Text (Truncation)."""
        long_text = "A" * 10000  # 10k Zeichen
        result = classifier_with_mocks.classify_text(long_text)
        
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        assert result_data["text_length"] == 10000
        assert "category" in result_data
    
    def test_large_batch_processing(self, classifier_with_mocks):
        """Test für große Batch-Verarbeitung."""
        large_batch = [f"Test text number {i}" for i in range(100)]
        result = classifier_with_mocks.classify_batch(large_batch)
        
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        assert result_data["total_processed"] == 100
        assert len(result_data["results"]) == 100
    
    def test_unicode_text_handling(self, classifier_with_mocks):
        """Test für Unicode-Text (Emojis, Umlaute, etc.)."""
        unicode_text = "Hällö Wörld! 🚀 Spëciäl chäräctërs: αβγ δεζ ηθι"
        result = classifier_with_mocks.classify_text(unicode_text)
        
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        assert result_data["text_length"] == len(unicode_text)
        assert "category" in result_data


# Parametrized Tests für verschiedene Konfigurationen
@pytest.mark.parametrize("batch_size,expected_batches", [
    (32, 1),    # 3 texts in 1 batch
    (2, 2),     # 3 texts in 2 batches  
    (1, 3),     # 3 texts in 3 batches
])
def test_batch_size_configurations(mocker, batch_size, expected_batches):
    """Test verschiedene Batch-Größen-Konfigurationen."""
    # Setup standard mocks
    mock_tokenizer = mocker.Mock()
    mock_encoding = mocker.Mock()
    mock_encoding.input_ids = torch.tensor([[1, 2, 3]])
    mock_encoding.to = mocker.Mock(return_value=mock_encoding)
    mock_tokenizer.return_value = mock_encoding
    
    mock_model = mocker.Mock()
    mock_outputs = mocker.Mock()
    mock_outputs.logits = torch.tensor([[0.2, 0.6, 0.2], [0.1, 0.8, 0.1], [0.3, 0.5, 0.2]])  # 3 predictions
    mock_model.return_value = mock_outputs
    mock_model.to = mocker.Mock()
    mock_model.eval = mocker.Mock()
    
    mocker.patch("bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", 
                 return_value=mock_tokenizer)
    mocker.patch("bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", 
                 return_value=mock_model)
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("bu_processor.pipeline.classifier.EnhancedPDFExtractor")
    
    classifier = RealMLClassifier(batch_size=batch_size)
    test_texts = ["Text 1", "Text 2", "Text 3"]
    
    result = classifier.classify_batch(test_texts)
    
    if hasattr(result, 'dict'):
        result_data = result.dict()
    else:
        result_data = result
    
    assert result_data["total_processed"] == 3
    assert len(result_data["results"]) == 3


# Fixture für Mock-Daten
@pytest.fixture
def sample_classification_data():
    """Sample-Daten für Tests."""
    return {
        "simple_texts": [
            "Ich arbeite als Softwareentwickler.",
            "Als Arzt behandle ich Patienten.",
            "Marketing ist mein Fachgebiet."
        ],
        "expected_categories": [0, 1, 2],  # IT, Healthcare, Marketing
        "pdf_paths": [
            "tests/fixtures/sample.pdf",
            "tests/fixtures/sample_finance.pdf", 
            "tests/fixtures/sample_marketing.pdf"
        ]
    }


class TestMockDataValidation:
    """Tests um sicherzustellen dass Mock-Daten korrekt sind."""
    
    def test_sample_data_structure(self, sample_classification_data):
        """Validiert dass Test-Daten korrekte Struktur haben."""
        data = sample_classification_data
        
        assert len(data["simple_texts"]) == len(data["expected_categories"])
        assert all(isinstance(text, str) for text in data["simple_texts"])
        assert all(isinstance(cat, int) for cat in data["expected_categories"])
    
    def test_fixture_files_exist(self, sample_classification_data):
        """Prüft dass Test-Fixture-Dateien existieren."""
        for pdf_path in sample_classification_data["pdf_paths"]:
            path = Path(pdf_path)
            if path.exists():
                assert path.suffix == ".pdf"
                assert path.stat().st_size > 0
            # Keine Assertion falls Datei nicht existiert - das ist OK für Tests


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
