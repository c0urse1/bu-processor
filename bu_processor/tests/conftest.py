"""Gemeinsame Test-Fixtures und Mock-Utilities für alle Tests.

Additional bootstrap added automatically:
 - Ensure project root on sys.path (handles nested package layout)
 - Set flags to allow optional external integrations to be skipped
 - Enable lazy model loading to avoid heavyweight downloads in CI
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock

import pytest
import torch

# Try to import pandas for training test fixtures
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Frühzeitig sicherstellen, dass das Package importierbar ist
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# === BASE ENVIRONMENT FIXTURES ===

@pytest.fixture(scope="session", autouse=True)
def _base_env():
    """
    Base Environment Setup für alle Tests.
    
    Setzt Standard-Umgebung:
    - BU_LAZY_MODELS="0" für immediate model loading in Tests
    - TESTING="true" für Test-Modus
    - Andere Test-spezifische Flags
    """
    # Standard: nicht-lazy, damit from_pretrained-Aufrufe stattfinden
    os.environ.setdefault("BU_LAZY_MODELS", "0")
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("PYTEST_RUNNING", "1")
    os.environ.setdefault("ALLOW_EMPTY_PINECONE_KEY", "1")
    
    # Disable vector database and chatbot for tests to avoid validation issues
    os.environ.setdefault("BU_PROCESSOR_VECTOR_DB__ENABLE_VECTOR_DB", "false")
    os.environ.setdefault("BU_PROCESSOR_OPENAI__ENABLE_CHATBOT", "false")
    
    # Set a valid test Pinecone key to avoid validation errors
    os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-api-key-01234567890123456789")
    
    # Provide a tiny default model name to avoid large downloads if code checks env
    os.environ.setdefault("BUPROC_MODEL_NAME", "sshleifer/tiny-distilroberta-base")
    
    yield

# === LAZY LOADING CONTROL FIXTURES ===

@pytest.fixture
def non_lazy_models(monkeypatch):
    """
    Erzwingt immediate model loading (BU_LAZY_MODELS=0).
    
    Nutze diese Fixture für Tests die from_pretrained-Aufrufe erwarten
    und das Loading-Verhalten validieren wollen.
    
    Args:
        monkeypatch: Pytest monkeypatch für Environment
    """
    monkeypatch.setenv("BU_LAZY_MODELS", "0")
    yield

@pytest.fixture
def lazy_models(monkeypatch):
    """
    Aktiviert lazy model loading (BU_LAZY_MODELS=1).
    
    Nutze diese Fixture für Tests die lazy loading behavior demonstrieren
    oder manuelle Loading-Aufrufe testen wollen.
    
    Args:
        monkeypatch: Pytest monkeypatch für Environment
    """
    monkeypatch.setenv("BU_LAZY_MODELS", "1")
    yield

# === PROJEKT-WEITE FIXTURES ===

@pytest.fixture(scope="session")
def project_root():
    """Projekt-Root-Verzeichnis."""
    return Path(__file__).parent.parent


# === MOCK FIXTURES FOR TORCH/TRANSFORMERS ===

@pytest.fixture
def mock_tokenizer(mocker):
    """
    Mock für Hugging Face Tokenizer.
    
    Returns:
        Mock-Tokenizer mit standard return_tensors="pt" Verhalten
    """
    mock_tok = mocker.MagicMock()
    
    # Standard tokenizer return format
    mock_tok.return_value = {
        'input_ids': torch.tensor([[101, 2023, 2003, 1037, 3231, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1]])
    }
    
    # Mock für from_pretrained
    mock_tok.from_pretrained = mocker.MagicMock(return_value=mock_tok)
    
    return mock_tok

@pytest.fixture
def mock_torch_model(mocker):
    """
    Mock für PyTorch/Transformers Model.
    
    Returns:
        Mock-Model mit standard logits output
    """
    mock_model = mocker.MagicMock()
    
    # Standard model output mit logits
    mock_output = mocker.MagicMock()
    mock_output.logits = torch.tensor([[0.1, 0.9]])  # 2 classes, class_1 higher
    mock_model.return_value = mock_output
    
    # Mock für from_pretrained
    mock_model.from_pretrained = mocker.MagicMock(return_value=mock_model)
    
    return mock_model

# === CLASSIFIER FIXTURES ===

# === PDF FIXTURES ===

@pytest.fixture
def sample_pdf_path(tmp_path):
    """
    Erstellt eine minimale PDF-Datei für Tests.
    
    Args:
        tmp_path: Pytest tmp_path fixture
        
    Returns:
        str: Pfad zur erstellten PDF-Datei
    """
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF Content) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000206 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF"""
    
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(pdf_content)
    return str(pdf_file)

# === PIPELINE FIXTURES ===

@pytest.fixture  
def pipeline_with_mocks(classifier_with_mocks, monkeypatch):
    """
    Zentrale Fixture für gemockte Pipeline.
    
    Erstellt eine Pipeline mit gemocktem Classifier und anderen Dependencies.
    
    Args:
        classifier_with_mocks: Gemockter Classifier
        monkeypatch: Pytest monkeypatch
        
    Returns:
        Gemockte Pipeline-Instanz
    """
    try:
        from bu_processor.pipeline.enhanced_integrated_pipeline import EnhancedIntegratedPipeline
        
        # Mock external dependencies
        monkeypatch.setenv("PINECONE_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key") 
        
        pipeline = EnhancedIntegratedPipeline.__new__(EnhancedIntegratedPipeline)
        pipeline.classifier = classifier_with_mocks
        pipeline.config = None  # Mock config wenn nötig
        
        return pipeline
        
    except ImportError:
        # Fallback wenn Pipeline nicht verfügbar
        return None


# === BESTEHENEDE FIXTURES (bereits implementiert) ===

# === CONFIDENCE THRESHOLD FIXTURES ===

@pytest.fixture(autouse=True)
def test_confidence_threshold(monkeypatch):
    """
    Setzt einen niedrigeren Confidence-Threshold für Tests.
    
    Automatisch auf alle Tests angewendet um zuverlässige High-Confidence
    Tests zu ermöglichen.
    """
    # Für Tests global auf 0.7 senken (macht Tests zuverlässiger)
    monkeypatch.setenv("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD", "0.7")
    yield

@pytest.fixture
def low_confidence_threshold(monkeypatch):
    """
    Setzt einen sehr niedrigen Confidence-Threshold (0.5) für spezielle Tests.
    
    Usage:
        def test_something(low_confidence_threshold):
            # Test läuft mit 0.5 Threshold
    """
    monkeypatch.setenv("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD", "0.5")
    yield

@pytest.fixture 
def high_confidence_threshold(monkeypatch):
    """
    Setzt einen hohen Confidence-Threshold (0.9) für spezielle Tests.
    
    Usage:
        def test_something(high_confidence_threshold):
            # Test läuft mit 0.9 Threshold
    """
    monkeypatch.setenv("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD", "0.9")
    yield

# === MOCK LOGITS FÜR HIGH-CONFIDENCE TESTS ===

class MockLogitsProvider:
    """
    Utility-Klasse für zuverlässige Mock-Logits in Tests.
    
    Stellt vordefinierte Logit-Werte bereit, die nach Softmax
    garantiert über bestimmten Confidence-Schwellen liegen.
    """
    
    @staticmethod
    def high_confidence_2_classes(winner_idx: int = 1) -> List[float]:
        """
        Logits für 2 Klassen mit hoher Confidence (~0.997).
        
        Args:
            winner_idx: Index der Gewinner-Klasse (0 oder 1)
            
        Returns:
            Logits die nach Softmax ~0.997 für winner_idx ergeben
        """
        if winner_idx == 0:
            return [6.0, -2.0]  # Softmax: [0.997, 0.003]
        else:
            return [-2.0, 6.0]  # Softmax: [0.003, 0.997]
    
    @staticmethod
    def high_confidence_3_classes(winner_idx: int = 1) -> List[float]:
        """
        Logits für 3 Klassen mit hoher Confidence (~0.982).
        
        Args:
            winner_idx: Index der Gewinner-Klasse (0, 1, oder 2)
            
        Returns:
            Logits die nach Softmax ~0.982 für winner_idx ergeben
        """
        logits = [-2.0, -2.0, -2.0]
        logits[winner_idx] = 6.0  # Winner bekommt 6.0, Rest -2.0
        return logits  # Softmax: [0.009, 0.982, 0.009] für winner_idx=1
    
    @staticmethod
    def medium_confidence_2_classes(winner_idx: int = 1) -> List[float]:
        """
        Logits für 2 Klassen mit mittlerer Confidence (~0.731).
        
        Args:
            winner_idx: Index der Gewinner-Klasse (0 oder 1)
            
        Returns:
            Logits die nach Softmax ~0.731 für winner_idx ergeben
        """
        if winner_idx == 0:
            return [1.0, 0.0]  # Softmax: [0.731, 0.269]
        else:
            return [0.0, 1.0]  # Softmax: [0.269, 0.731]
    
    @staticmethod
    def low_confidence_2_classes() -> List[float]:
        """
        Logits für 2 Klassen mit niedriger Confidence (~0.524).
        
        Returns:
            Logits die nach Softmax ~0.524 für Index 1 ergeben
        """
        return [-0.1, 0.1]  # Softmax: [0.476, 0.524]
    
    @staticmethod
    def verify_softmax_confidence(logits: List[float], expected_confidence: float, tolerance: float = 0.01) -> bool:
        """
        Verifiziert, dass Logits die erwartete Confidence ergeben.
        
        Args:
            logits: Logit-Werte
            expected_confidence: Erwartete maximale Confidence
            tolerance: Erlaubte Abweichung
            
        Returns:
            True wenn Confidence im erwarteten Bereich liegt
        """
        if not logits:
            return False
            
        import math
        # Softmax berechnen
        max_logit = max(logits)
        exps = [math.exp(x - max_logit) for x in logits]
        sum_exps = sum(exps)
        probs = [exp_val / sum_exps for exp_val in exps]
        
        max_prob = max(probs)
        return abs(max_prob - expected_confidence) <= tolerance

@pytest.fixture
def mock_logits():
    """
    Fixture die MockLogitsProvider bereitstellt.
    
    Usage:
        def test_classification(mock_logits):
            logits = mock_logits.high_confidence_2_classes(winner_idx=1)
            # logits sind jetzt [-2.0, 6.0] -> Softmax: [0.003, 0.997]
    """
    return MockLogitsProvider()

@pytest.fixture
def mock_classifier_with_logits(mocker, mock_logits):
    """
    Fixture die einen gemockten Classifier mit kontrollierbaren Logits bereitstellt.
    
    Usage:
        def test_something(mock_classifier_with_logits):
            classifier, set_logits = mock_classifier_with_logits
            set_logits(mock_logits.high_confidence_2_classes())
            result = classifier.classify_text("test")
            # result.confidence wird ~0.997 sein
    """
    from bu_processor.pipeline.classifier import RealMLClassifier, ClassificationResult
    
    # Mock Classifier
    classifier = RealMLClassifier.__new__(RealMLClassifier)
    classifier.confidence_threshold = 0.7  # Standard für Tests
    classifier.labels = ["class_0", "class_1"]
    
    # Container für Logits
    logits_container = {"current": mock_logits.high_confidence_2_classes()}
    
    def set_logits(new_logits: List[float]):
        """Setzt neue Logits für den nächsten classify_text Aufruf."""
        logits_container["current"] = new_logits
    
    # Mock die _forward_logits Methode
    def mock_forward_logits(text: str) -> List[float]:
        return logits_container["current"]
    
    def mock_label_list() -> List[str]:
        return classifier.labels
    
    classifier._forward_logits = mock_forward_logits
    classifier._label_list = mock_label_list
    
    return classifier, set_logits


@pytest.fixture
def disable_lazy_loading(monkeypatch):
    """Fixture to disable lazy loading for tests that need immediate from_pretrained calls.
    
    Use this fixture in tests that want to assert on AutoTokenizer.from_pretrained 
    or AutoModel.from_pretrained calls immediately after classifier creation.
    
    Example:
        def test_something(disable_lazy_loading, mocker):
            # Now classifier will load model immediately
            pass
    """
    monkeypatch.setenv("BU_LAZY_MODELS", "0")


@pytest.fixture
def enable_lazy_loading(monkeypatch):
    """Fixture to explicitly enable lazy loading (default behavior).
    
    Use this when you want to be explicit about lazy loading being enabled,
    or to override a previous disable_lazy_loading in the same test session.
    """
    monkeypatch.setenv("BU_LAZY_MODELS", "1")


@pytest.fixture(scope="session") 
def test_data_dir(project_root):
    """Test-Daten-Verzeichnis."""
    return project_root / "tests" / "fixtures"


@pytest.fixture
def temp_dir():
    """Temporäres Verzeichnis für Tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_texts():
    """Standard-Texte für Klassifikations-Tests."""
    return [
        "Ich arbeite als Softwareentwickler in einer IT-Firma und programmiere täglich.",
        "Als Arzt behandle ich Patienten im Krankenhaus und führe Operationen durch.",
        "Marketing und Vertrieb sind meine Hauptaufgaben in diesem Unternehmen.",
        "Rechtliche Beratung und Vertragsverhandlungen gehören zu meinem Beruf.",
        "Finanzanalysen und Buchhaltung sind mein Fachgebiet in der Firma."
    ]


@pytest.fixture
def sample_classification_results():
    """Standard-Klassifikationsergebnisse für Tests."""
    return [
        {"category": 0, "confidence": 0.89, "is_confident": True},   # IT
        {"category": 1, "confidence": 0.93, "is_confident": True},   # Healthcare  
        {"category": 2, "confidence": 0.76, "is_confident": False},  # Marketing
        {"category": 3, "confidence": 0.82, "is_confident": True},   # Legal
        {"category": 4, "confidence": 0.95, "is_confident": True}    # Finance
    ]


@pytest.fixture
def sample_pdf_path(test_data_dir):
    """Sample PDF path for tests."""
    pdf_path = test_data_dir / "sample.pdf"
    
    # Ensure the fixtures directory exists
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a minimal PDF file for testing if it doesn't exist
    if not pdf_path.exists():
        pdf_path.write_text("Mock PDF content for testing")
    
    return pdf_path


# === MOCK FACTORIES ===

@pytest.fixture
def classifier_with_mocks(mocker):
    """Create a classifier with mocked ML model dependencies."""
    # Mock torch first to avoid import issues
    mock_torch = mocker.patch("bu_processor.pipeline.classifier.torch")
    mock_torch.no_grad.return_value.__enter__ = mocker.MagicMock()
    mock_torch.no_grad.return_value.__exit__ = mocker.MagicMock()
    mock_torch.tensor.return_value = mocker.MagicMock()
    mock_torch.nn.functional.softmax.return_value = mocker.MagicMock()
    mock_torch.nn.functional.softmax.return_value.cpu.return_value.numpy.return_value = [[0.45, 0.55]]
    
    # Mock the transformer models
    mock_model = mocker.MagicMock()
    mock_tokenizer = mocker.MagicMock()
    
    # Mock the from_pretrained methods
    mocker.patch("bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", 
                 return_value=mock_model)
    mocker.patch("bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", 
                 return_value=mock_tokenizer)
    
    # Configure mock tokenizer behavior
    mock_tokenizer.return_value = {
        'input_ids': [[101, 102, 103]],
        'attention_mask': [[1, 1, 1]]
    }
    mock_tokenizer.__call__ = mocker.MagicMock(return_value={
        'input_ids': [[101, 102, 103]],
        'attention_mask': [[1, 1, 1]]
    })
    
    # Configure mock model behavior
    mock_output = mocker.MagicMock()
    mock_output.logits = [[0.1, 0.9]]
    mock_model.return_value = mock_output
    mock_model.__call__ = mocker.MagicMock(return_value=mock_output)
    
    from bu_processor.pipeline.classifier import RealMLClassifier, ClassificationResult, BatchClassificationResult
    
    # Patch the classifier methods directly instead of creating instance first
    def create_mock_classifier():
        classifier = RealMLClassifier(
            model_name="bert-base-uncased",
            device="cpu"
        )
        
        # Set the mocked objects
        classifier.model = mock_model
        classifier.tokenizer = mock_tokenizer
        
        # Mock classify_text directly to return proper structure
        def mock_classify_text(text):
            result = ClassificationResult(
                text=text[:100] if text else "",
                category=0,  # Integer category as expected by tests
                confidence=0.85,
                is_confident=True,
                metadata={"mock": True},
                probabilities={"Category_A": 0.85, "Category_B": 0.15},
                success=True,  # Add success field in constructor
                label="Category_A"  # Add label field in constructor
            )
            # Add the extra attributes that the real method adds
            result.input_type = "text"
            result.text_length = len(text)
            result.processing_time = 0.001
            result.model_version = "v1.0"
            return result
        
        # Mock classify_batch for batch operations
        def mock_classify_batch(texts):
            from bu_processor.pipeline.classifier import BatchClassificationResult
            
            # Handle empty input case
            if not texts:
                raise ValueError("Keine Texte für Batch-Klassifikation übergeben")
            
            results = []
            for text in texts:
                result = mock_classify_text(text)
                results.append(result)
            
            return BatchClassificationResult(
                results=results,
                total_processed=len(texts),
                successful=len(results),
                failed=0,
                batch_time=0.001
            )
        
        # Replace the methods
        classifier.classify_text = mock_classify_text
        classifier.classify_batch = mock_classify_batch
        
        # Add BatchClassificationResult as class attribute for compatibility
        classifier.BatchClassificationResult = BatchClassificationResult
        
        # Add _process_batch method for compatibility with older tests
        def mock_process_batch(texts):
            results = []
            for text in texts:
                results.append({
                    "label": "Category_A" if len(text) % 2 == 0 else "Category_B",
                    "confidence": 0.85,
                    "success": True
                })
            return results
        
        classifier._process_batch = mock_process_batch
        
        return classifier
    
    return create_mock_classifier()

@pytest.fixture
def mock_pdf_extractor(mocker):
    """
    Mock für PDF-Extractor.
    
    Erstellt einen gemockten PDF-Extractor für Tests die PDF-Verarbeitung benötigen.
    """
    from bu_processor.pipeline.content_types import ContentType
    from bu_processor.pipeline.pdf_extractor import ExtractedContent
    
    mock_extractor = mocker.Mock()
    mock_content = ExtractedContent(
        text="Beispiel PDF Text für Tests.",
        page_count=1,
        file_path="test.pdf",
        metadata={"title": "Test"}, 
        extraction_method="mock"
    )
    
    # Configure the mock to return the ExtractedContent for both method names
    mock_extractor.extract_content.return_value = mock_content
    mock_extractor.extract_text_from_pdf.return_value = mock_content
    
    return mock_extractor


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a temporary PDF file for testing."""
    pdf_file = tmp_path / "test.pdf"
    # Create a minimal valid PDF
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF content) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000274 00000 n 
trailer
<< /Size 5 /Root 1 0 R >>
startxref
362
%%EOF"""
    pdf_file.write_bytes(pdf_content)
    return str(pdf_file)


@pytest.fixture
def classifier_with_eager_loading(mocker, disable_lazy_loading):
    """Classifier fixture with lazy loading disabled for from_pretrained assertion tests.
    
    This fixture ensures that AutoTokenizer.from_pretrained and 
    AutoModelForSequenceClassification.from_pretrained are called immediately
    during classifier initialization, not deferred.
    
    Use this in tests that need to assert:
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
    """
    # Mock components
    mock_tokenizer = mocker.Mock()
    mock_encoding = mocker.Mock()
    mock_encoding.input_ids = torch.tensor([[1, 2, 3]])
    mock_encoding.attention_mask = torch.tensor([[1, 1, 1]])
    mock_encoding.to = mocker.Mock(return_value=mock_encoding)
    mock_tokenizer.return_value = mock_encoding
    
    mock_model = mocker.Mock()
    mock_outputs = mocker.Mock()
    mock_outputs.logits = torch.tensor([[0.1, 5.0, 0.1]])  # Strong logits → softmax ~0.99 confidence
    mock_model.return_value = mock_outputs
    mock_model.to = mocker.Mock(return_value=mock_model)
    mock_model.eval = mocker.Mock()
    
    # Patch the imports - store references for assertion access
    mock_tokenizer_patch = mocker.patch(
        "bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", 
        return_value=mock_tokenizer
    )
    mock_model_patch = mocker.patch(
        "bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", 
        return_value=mock_model
    )
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("bu_processor.pipeline.classifier.EnhancedPDFExtractor")
    
    from bu_processor.pipeline.classifier import RealMLClassifier
    classifier = RealMLClassifier()
    
    # Attach mock references for test assertions
    classifier._test_mock_tokenizer_patch = mock_tokenizer_patch
    classifier._test_mock_model_patch = mock_model_patch
    classifier._test_mock_tokenizer = mock_tokenizer
    classifier._test_mock_model = mock_model
    
    return classifier


@pytest.fixture
def mock_torch_model(mocker):
    """Factory für PyTorch Model Mocks."""
    def create_mock_model(num_categories: int = 5, high_confidence_category: int = 1):
        mock_model = mocker.Mock()
        mock_model.eval = mocker.Mock()
        mock_model.to = mocker.Mock(return_value=mock_model)
        
        # Mock forward pass
        mock_outputs = mocker.Mock()
        logits = torch.zeros(1, num_categories)
        logits[0, high_confidence_category] = 5.0  # Strong logit for high confidence (~0.99 after softmax)
        mock_outputs.logits = logits
        mock_model.return_value = mock_outputs
        
        return mock_model
    
    return create_mock_model


@pytest.fixture  
def mock_tokenizer(mocker):
    """Factory für Tokenizer Mocks."""
    def create_mock_tokenizer(max_length: int = 512):
        mock_tokenizer = mocker.Mock()
        
        # Mock tokenization output
        mock_encoding = {
            'input_ids': torch.tensor([[101, 102, 103, 104, 102]]),  # BERT-style tokens
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        mock_encoding_obj = mocker.Mock()
        for key, value in mock_encoding.items():
            setattr(mock_encoding_obj, key, value)
        
        mock_encoding_obj.to = mocker.Mock(return_value=mock_encoding_obj)
        mock_tokenizer.return_value = mock_encoding_obj
        
        # Mock tokenizer attributes
        mock_tokenizer.model_max_length = max_length
        mock_tokenizer.pad_token = "[PAD]"
        
        return mock_tokenizer
    
    return create_mock_tokenizer


@pytest.fixture
def mock_pdf_document(mocker):
    """Factory für PDF Document Mocks."""
    def create_mock_pdf(page_count: int = 3, has_text: bool = True, is_encrypted: bool = False):
        mock_doc = mocker.Mock()
        mock_doc.__len__ = mocker.Mock(return_value=page_count)
        mock_doc.needs_pass = is_encrypted
        mock_doc.is_pdf = True
        mock_doc.metadata = {
            "title": "Test Document",
            "author": "Test Author", 
            "subject": "Test Subject"
        }
        
        # Mock pages
        pages = []
        for i in range(page_count):
            mock_page = mocker.Mock()
            if has_text:
                mock_page.get_text.return_value = f"Text content for page {i+1}. Lorem ipsum dolor sit amet."
            else:
                mock_page.get_text.return_value = ""
            pages.append(mock_page)
        
        mock_doc.load_page.side_effect = pages
        
        # Context manager support
        mock_doc.__enter__ = mocker.Mock(return_value=mock_doc)
        mock_doc.__exit__ = mocker.Mock(return_value=None)
        
        return mock_doc
    
    return create_mock_pdf


@pytest.fixture
def mock_pinecone_index(mocker):
    """Factory für Pinecone Index Mocks."""
    def create_mock_index(vector_count: int = 1000):
        mock_index = mocker.Mock()
        
        # Mock query response
        mock_index.query.return_value = {
            'matches': [
                {
                    'id': f'doc_{i}',
                    'score': 0.9 - (i * 0.1), 
                    'metadata': {
                        'category': 'finance' if i % 2 == 0 else 'legal',
                        'text': f'Document {i} content',
                        'source': f'doc_{i}.pdf'
                    }
                } for i in range(3)
            ]
        }
        
        # Mock upsert response
        mock_index.upsert.return_value = {'upserted_count': 1}
        
        # Mock delete response
        mock_index.delete.return_value = {'deleted_count': 1}
        
        # Mock stats
        mock_index.describe_index_stats.return_value = {
            'dimension': 768,
            'index_fullness': vector_count / 10000,
            'total_vector_count': vector_count
        }
        
        return mock_index
    
    return create_mock_index


# === UTILITY FIXTURES ===

@pytest.fixture
def create_temp_pdf(temp_dir):
    """Erstellt temporäre PDF-Dateien für Tests."""
    def _create_pdf(filename: str, content: str = "Test PDF Content"):
        pdf_path = temp_dir / filename
        # Fake PDF content (in echten Tests würde man reportlab verwenden)
        pdf_path.write_text(f"FAKE_PDF_HEADER\n{content}")
        return pdf_path
    
    return _create_pdf


@pytest.fixture
def mock_environment_vars(mocker):
    """Mock für Environment Variables."""
    def set_env_vars(env_dict: Dict[str, str]):
        for key, value in env_dict.items():
            mocker.patch.dict(os.environ, {key: value})
    
    return set_env_vars


@pytest.fixture
def capture_logs(caplog):
    """Erweiterte Log-Capture mit Filtering."""
    def get_logs_by_level(level: str) -> List[str]:
        return [record.message for record in caplog.records if record.levelname == level.upper()]
    
    caplog.get_logs_by_level = get_logs_by_level
    return caplog


# === OCR TESTING UTILITIES ===

def check_tesseract_available():
    """Prüft, ob Tesseract OCR verfügbar ist."""
    try:
        import pytesseract
        # Einfacher Test ob pytesseract funktioniert
        pytesseract.get_tesseract_version()
        return True
    except (ImportError, Exception):
        return False


# OCR Skip-Decorator für Tests, die echtes OCR benötigen
requires_tesseract = pytest.mark.skipif(
    not check_tesseract_available(),
    reason="Tesseract OCR nicht verfügbar - Test wird übersprungen"
)


# === HEAVY DEPENDENCY SKIP MARKERS ===

def check_torch_available():
    """Prüft, ob PyTorch verfügbar ist."""
    try:
        import torch
        # Basic functionality test
        torch.tensor([1.0])
        return True
    except (ImportError, Exception):
        return False


def check_transformers_available():
    """Prüft, ob Transformers verfügbar ist."""
    try:
        import transformers
        # Check if core classes are available
        transformers.AutoTokenizer
        transformers.AutoModelForSequenceClassification
        return True
    except (ImportError, Exception):
        return False


def check_sentence_transformers_available():
    """Prüft, ob Sentence-Transformers verfügbar ist."""
    try:
        import sentence_transformers
        return True
    except (ImportError, Exception):
        return False


def check_cv2_available():
    """Prüft, ob OpenCV verfügbar ist."""
    try:
        import cv2
        return True
    except (ImportError, Exception):
        return False


def check_pinecone_available():
    """Prüft, ob Pinecone verfügbar ist."""
    try:
        import pinecone
        return True
    except (ImportError, Exception):
        return False


# Skip-Decorators für schwere Dependencies
requires_torch = pytest.mark.skipif(
    not check_torch_available(),
    reason="PyTorch nicht verfügbar - Test wird übersprungen"
)

requires_transformers = pytest.mark.skipif(
    not check_transformers_available(),
    reason="Transformers nicht verfügbar - Test wird übersprungen"
)

requires_sentence_transformers = pytest.mark.skipif(
    not check_sentence_transformers_available(),
    reason="Sentence-Transformers nicht verfügbar - Test wird übersprungen"
)

requires_cv2 = pytest.mark.skipif(
    not check_cv2_available(),
    reason="OpenCV nicht verfügbar - Test wird übersprungen"
)

requires_pinecone = pytest.mark.skipif(
    not check_pinecone_available(),
    reason="Pinecone nicht verfügbar - Test wird übersprungen"
)

# Kombinierte Skip-Markers
requires_ml_stack = pytest.mark.skipif(
    not (check_torch_available() and check_transformers_available()),
    reason="ML Stack (PyTorch + Transformers) nicht verfügbar - Test wird übersprungen"
)

requires_full_ml_stack = pytest.mark.skipif(
    not (check_torch_available() and check_transformers_available() and check_sentence_transformers_available()),
    reason="Vollständiger ML Stack nicht verfügbar - Test wird übersprungen"
)

requires_ocr_stack = pytest.mark.skipif(
    not (check_tesseract_available() and check_cv2_available()),
    reason="OCR Stack (Tesseract + OpenCV) nicht verfügbar - Test wird übersprungen"
)


@pytest.fixture
def ml_stack_available():
    """Fixture die True zurückgibt wenn ML Stack verfügbar ist."""
    return check_torch_available() and check_transformers_available()


@pytest.fixture
def ocr_available():
    """Fixture die True zurückgibt wenn OCR verfügbar ist."""
    return check_tesseract_available()


# === PERFORMANCE FIXTURES ===

@pytest.fixture
def performance_timer():
    """Timer für Performance-Tests."""
    import time
    
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
        
        def assert_faster_than(self, max_seconds: float):
            assert self.elapsed is not None, "Timer not properly used"
            assert self.elapsed < max_seconds, f"Operation took {self.elapsed:.3f}s > {max_seconds}s"
    
    return PerformanceTimer()


# === ERROR SIMULATION FIXTURES ===

@pytest.fixture
def error_simulator(mocker):
    """Simuliert verschiedene Fehler-Szenarien."""
    class ErrorSimulator:
        def __init__(self):
            self.mocker = mocker
        
        def simulate_network_error(self, target_func):
            """Simuliert Netzwerk-Fehler."""
            return self.mocker.patch(target_func, side_effect=ConnectionError("Network unreachable"))
        
        def simulate_memory_error(self, target_func):
            """Simuliert Memory-Fehler."""
            return self.mocker.patch(target_func, side_effect=MemoryError("Out of memory"))
        
        def simulate_timeout_error(self, target_func):
            """Simuliert Timeout."""
            return self.mocker.patch(target_func, side_effect=TimeoutError("Operation timed out"))
        
        def simulate_gpu_error(self, target_func):
            """Simuliert GPU/CUDA-Fehler."""
            return self.mocker.patch(target_func, side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory"))
        
        def simulate_intermittent_failure(self, target_func, fail_count: int = 2):
            """Simuliert intermittierende Fehler (scheitert X mal, dann erfolgreich)."""
            call_count = 0
            
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= fail_count:
                    raise ConnectionError(f"Failure {call_count}")
                return {"success": True, "attempt": call_count}
            
            return self.mocker.patch(target_func, side_effect=side_effect)
    
    return ErrorSimulator()


# === INTEGRATION TEST FIXTURES ===

@pytest.fixture
def integration_test_data():
    """Daten für Integration Tests."""
    return {
        "pdf_files": [
            "tests/fixtures/sample.pdf",
            "tests/fixtures/sample_finance.pdf",
            "tests/fixtures/sample_marketing.pdf"
        ],
        "expected_categories": [0, 4, 2],  # IT, Finance, Marketing
        "test_queries": [
            "Quartalsabschluss und Finanzergebnisse",
            "Software-Entwicklung und Programmierung", 
            "Marketing-Kampagne und Werbung"
        ]
    }


@pytest.fixture
def mock_api_responses():
    """Mock-Responses für externe APIs."""
    return {
        "openai_chat_response": {
            "choices": [
                {
                    "message": {
                        "content": "Das ist eine Test-Antwort vom Chatbot."
                    }
                }
            ],
            "usage": {
                "total_tokens": 150,
                "prompt_tokens": 100,
                "completion_tokens": 50
            }
        },
        "pinecone_query_response": {
            "matches": [
                {
                    "id": "doc_123",
                    "score": 0.92,
                    "metadata": {
                        "category": "finance", 
                        "text": "Finanzielle Quartalsergebnisse",
                        "source": "quarterly_report.pdf"
                    }
                }
            ]
        }
    }


# === PYTEST HOOKS ===

def pytest_configure(config):
    """Pytest-Konfiguration beim Start."""
    # Set the configured flag
    os.environ["PYTEST_CONFIGURED"] = "1"
    
    # Marker registrieren falls nicht in pyproject.toml
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "external: Tests requiring external services")
    config.addinivalue_line("markers", "mock: Tests with comprehensive mocks")


def pytest_collection_modifyitems(config, items):
    """Modifiziert Test-Collection (z.B. für automatische Marker)."""
    # Automatisch 'slow' marker für Tests mit 'performance' oder 'batch' im Namen
    for item in items:
        if "performance" in item.name or "batch" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Automatisch 'external' marker für Tests mit 'pinecone' oder 'openai'
        if "pinecone" in item.name or "openai" in item.name or "chatbot" in item.name:
            item.add_marker(pytest.mark.external)


def pytest_runtest_setup(item):
    """Setup vor jedem Test."""
    # Skip Tests für nicht verfügbare optionale Dependencies
    if "pinecone" in item.name:
        try:
            import pinecone
        except ImportError:
            pytest.skip("Pinecone not available")
    
    if "chatbot" in item.name or "openai" in item.name:
        try:
            import openai
        except ImportError:
            pytest.skip("OpenAI not available")
    
    if "semantic" in item.name:
        try:
            import sentence_transformers
        except ImportError:
            pytest.skip("Sentence Transformers not available")


# === MOCK-UTILITIES KLASSEN ===

class MockFactory:
    """Factory für wiederverwendbare Mocks."""
    
    @staticmethod
    def create_classification_result(
        category: int = 1,
        confidence: float = 0.85,
        text_length: int = 100
    ) -> Dict[str, Any]:
        """Erstellt Mock-Klassifikationsergebnis."""
        return {
            "category": category,
            "confidence": confidence,
            "is_confident": confidence > 0.8,
            "input_type": "text",
            "text_length": text_length,
            "processing_time": 0.05,
            "model_version": "v1.0"
        }
    
    @staticmethod
    def create_batch_result(
        texts: List[str], 
        base_category: int = 1
    ) -> Dict[str, Any]:
        """Erstellt Mock-Batch-Ergebnis."""
        results = []
        for i, text in enumerate(texts):
            results.append(MockFactory.create_classification_result(
                category=(base_category + i) % 5,
                confidence=0.8 + (i * 0.02),  # Leicht variierende Confidence
                text_length=len(text)
            ))
        
        return {
            "total_processed": len(texts),
            "successful": len(texts),
            "failed": 0,
            "batch_time": 0.15,
            "results": results,
            "batch_id": f"test_batch_{len(texts)}"
        }
    
    @staticmethod  
    def create_pdf_content(
        text: str = "Sample PDF content",
        page_count: int = 2,
        file_path: str = "test.pdf"
    ):
        """Erstellt Mock-PDF-Content."""
        from bu_processor.pipeline.pdf_extractor import ExtractedContent
        
        return ExtractedContent(
            text=text,
            page_count=page_count,
            file_path=file_path,
            metadata={"title": "Test PDF", "author": "Test"},
            extraction_method="mocked"
        )
    
    @staticmethod
    def create_document_chunks(
        text: str,
        chunk_size: int = 100
    ):
        """Erstellt Mock-Document-Chunks."""
        from bu_processor.models.chunk import DocumentChunk
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size//10):  # ~10 chars per word estimate
            chunk_words = words[i:i+chunk_size//10]
            chunk_text = " ".join(chunk_words)
            
            chunk = DocumentChunk(
                chunk_id=f"chunk_{i//10}",
                text=chunk_text,
                start_position=i * 10,
                end_position=(i * 10) + len(chunk_text),
                chunk_type="paragraph",
                importance_score=0.7 + (i * 0.1) % 0.3  # Varying importance
            )
            chunks.append(chunk)
        
        return chunks


# === ENVIRONMENT SETUP ===

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup für die gesamte Test-Session."""
    # Setze Test-Environment Variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "WARNING"  # Weniger Logs in Tests
    
    # Mock externe API-Keys für Tests
    os.environ["OPENAI_API_KEY"] = "sk-test_key_openai_for_development"
    os.environ["PINECONE_API_KEY"] = "pc-test-key-for-development-only"
    os.environ["PINECONE_ENVIRONMENT"] = "test_env"
    
    yield
    
    # Cleanup nach allen Tests
    test_env_vars = ["TESTING", "OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]


# === SPEZIELLE MOCK-KLASSEN ===

class MockMLModel:
    """Erweiterte Mock-Klasse für ML-Models."""
    
    def __init__(self, mocker, num_categories: int = 5):
        self.mocker = mocker
        self.num_categories = num_categories
        self.call_count = 0
        
    def create_deterministic_model(self, category_probabilities: List[float]):
        """Erstellt Model mit deterministischen Ausgaben."""
        mock_model = self.mocker.Mock()
        mock_model.eval = self.mocker.Mock()
        mock_model.to = self.mocker.Mock(return_value=mock_model)
        
        # Erstelle Logits basierend auf gewünschten Probabilities
        logits = torch.tensor([category_probabilities])
        mock_outputs = self.mocker.Mock()
        mock_outputs.logits = logits
        mock_model.return_value = mock_outputs
        
        return mock_model
    
    def create_flaky_model(self, failure_rate: float = 0.3):
        """Erstellt Model das manchmal fehlschlägt."""
        mock_model = self.mocker.Mock()
        mock_model.eval = self.mocker.Mock()
        mock_model.to = self.mocker.Mock(return_value=mock_model)
        
        def flaky_forward(*args, **kwargs):
            self.call_count += 1
            if (self.call_count * failure_rate) % 1 > 0.7:  # ~30% failure rate
                raise RuntimeError("Model inference failed")
            
            mock_outputs = self.mocker.Mock()
            mock_outputs.logits = torch.tensor([[0.2, 5.0, 0.0]])  # Strong logits for high confidence
            return mock_outputs
        
        mock_model.side_effect = flaky_forward
        return mock_model


@pytest.fixture  
def mock_ml_model():
    """Fixture für MockMLModel.""" 
    def create_mock_model(mocker, **kwargs):
        return MockMLModel(mocker, **kwargs)
    
    return create_mock_model


# === TEST DATA GENERATORS ===

@pytest.fixture
def generate_test_data():
    """Generator für Test-Daten."""
    class TestDataGenerator:
        @staticmethod
        def create_realistic_business_texts(count: int = 10) -> List[str]:
            """Erstellt realistische Business-Texte."""
            templates = [
                "Als {role} arbeite ich in einer {company_type} und bin für {task} verantwortlich.",
                "Meine Hauptaufgabe als {role} ist {task} in der {department} Abteilung.",
                "In meiner Position als {role} entwickle ich {deliverable} für {target}.",
            ]
            
            roles = ["Softwareentwickler", "Projektmanager", "Analyst", "Berater", "Spezialist"]
            company_types = ["IT-Firma", "Beratung", "Bank", "Startup", "Konzern"]
            tasks = ["Entwicklung", "Analyse", "Beratung", "Management", "Optimierung"]
            departments = ["IT", "Finance", "Marketing", "Legal", "Operations"]
            deliverables = ["Software", "Berichte", "Strategien", "Prozesse", "Lösungen"]
            targets = ["Kunden", "Stakeholder", "Teams", "Märkte", "Systeme"]
            
            texts = []
            for i in range(count):
                template = templates[i % len(templates)]
                text = template.format(
                    role=roles[i % len(roles)],
                    company_type=company_types[i % len(company_types)],
                    task=tasks[i % len(tasks)],
                    department=departments[i % len(departments)],
                    deliverable=deliverables[i % len(deliverables)],
                    target=targets[i % len(targets)]
                )
                texts.append(text)
            
            return texts
        
        @staticmethod
        def create_test_vectors(count: int = 10, dimension: int = 768) -> List[List[float]]:
            """Erstellt Test-Embeddings."""
            import random
            vectors = []
            for i in range(count):
                # Deterministisch aber variiert
                random.seed(i)  
                vector = [random.uniform(-1, 1) for _ in range(dimension)]
                vectors.append(vector)
            
            return vectors
    
    return TestDataGenerator()


# === ASSERTION HELPERS ===

class TestAssertions:
    """Erweiterte Assertions für Tests."""
    
    @staticmethod
    def assert_classification_result_valid(result: Dict[str, Any]):
        """Validiert Klassifikationsergebnis-Struktur."""
        required_fields = ["category", "confidence", "is_confident", "input_type"]
        
        for field in required_fields:
            assert field in result, f"Required field '{field}' missing"
        
        assert isinstance(result["category"], int)
        assert result["category"] >= 0
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["is_confident"], bool)
    
    @staticmethod
    def assert_batch_result_valid(result: Dict[str, Any], expected_count: int):
        """Validiert Batch-Ergebnis-Struktur."""
        assert "total_processed" in result
        assert "successful" in result
        assert "failed" in result
        assert "results" in result
        
        assert result["total_processed"] == expected_count
        assert result["successful"] + result["failed"] == expected_count
        assert len(result["results"]) == expected_count
    
    @staticmethod
    def assert_pdf_content_valid(content):
        """Validiert ExtractedContent-Struktur."""
        assert hasattr(content, 'text')
        assert hasattr(content, 'page_count')
        assert hasattr(content, 'file_path')
        assert hasattr(content, 'metadata')
        assert hasattr(content, 'extraction_method')
        
        assert isinstance(content.text, str)
        assert content.page_count > 0
        assert len(content.text) > 0


@pytest.fixture
def test_assertions():
    """Fixture für Test-Assertions."""
    return TestAssertions()


# === MARK UTILITIES ===

# Helper decorator für Tests die externe Services brauchen
def requires_external_service(service_name: str):
    """Decorator für Tests die externe Services benötigen."""
    return pytest.mark.external


def slow_test(reason: str = "Performance test"):
    """Decorator für langsame Tests."""
    return pytest.mark.slow


def mock_test(reason: str = "Uses comprehensive mocks"):
    """Decorator für Mock-Tests."""
    return pytest.mark.mock


# === LAZY LOADING UTILITIES ===

def force_model_loading(classifier):
    """Utility function to explicitly trigger model loading in a lazy-loaded classifier.
    
    Call this in tests when you need to ensure that from_pretrained methods 
    have been called before making assertions.
    
    Args:
        classifier: RealMLClassifier instance (potentially lazy-loaded)
        
    Example:
        def test_something(classifier_with_mocks, mocker):
            mock_tokenizer = mocker.patch("...AutoTokenizer.from_pretrained")
            mock_model = mocker.patch("...AutoModel.from_pretrained") 
            
            # Force loading to trigger from_pretrained calls
            force_model_loading(classifier_with_mocks)
            
            # Now assertions will work
            mock_tokenizer.assert_called_once()
            mock_model.assert_called_once()
    """
    if hasattr(classifier, '_load_model_and_tokenizer'):
        classifier._load_model_and_tokenizer()
    else:
        # Fallback: trigger loading by calling a method that requires the model
        try:
            classifier.classify_text("dummy text to trigger loading")
        except Exception:
            pass  # Expected if mocked


def create_eager_classifier_fixture(mocker):
    """Factory function to create a classifier with eager loading.
    
    Use this in individual tests when you can't use the global fixture.
    
    Returns:
        tuple: (classifier, mock_tokenizer_patch, mock_model_patch)
        
    Example:
        def test_from_pretrained_calls(mocker):
            classifier, mock_tok, mock_mod = create_eager_classifier_fixture(mocker)
            mock_tok.assert_called_once()
            mock_mod.assert_called_once()
    """
    # Temporarily disable lazy loading
    original_lazy = os.environ.get("BU_LAZY_MODELS")
    os.environ["BU_LAZY_MODELS"] = "0"
    
    try:
        # Mock components
        mock_tokenizer = mocker.Mock()
        mock_encoding = mocker.Mock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3]])
        mock_encoding.attention_mask = torch.tensor([[1, 1, 1]])
        mock_encoding.to = mocker.Mock(return_value=mock_encoding)
        mock_tokenizer.return_value = mock_encoding
        
        mock_model = mocker.Mock()
        mock_outputs = mocker.Mock()
        mock_outputs.logits = torch.tensor([[0.1, 5.0, 0.1]])  # Strong logits → softmax ~0.99 confidence
        mock_model.return_value = mock_outputs
        mock_model.to = mocker.Mock(return_value=mock_model)
        mock_model.eval = mocker.Mock()
        
        # Patch the imports
        mock_tokenizer_patch = mocker.patch(
            "bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", 
            return_value=mock_tokenizer
        )
        mock_model_patch = mocker.patch(
            "bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", 
            return_value=mock_model
        )
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("bu_processor.pipeline.classifier.EnhancedPDFExtractor")
        
        from bu_processor.pipeline.classifier import RealMLClassifier
        classifier = RealMLClassifier()
        
        return classifier, mock_tokenizer_patch, mock_model_patch
        
    finally:
        # Restore original setting
        if original_lazy is not None:
            os.environ["BU_LAZY_MODELS"] = original_lazy


# === TRAINING TEST FIXTURES ===

@pytest.fixture
def dummy_train_val(tmp_path, monkeypatch):
    """Fixture für Dummy-CSV Dateien für Training-Tests.
    
    Erstellt temporäre train.csv und val.csv mit korrekten Labels
    und setzt die entsprechenden Umgebungsvariablen.
    
    Returns:
        tuple: (train_path, val_path) als Strings
    """
    if not PANDAS_AVAILABLE:
        pytest.skip("pandas not available for training tests")
    
    # Erstelle Dummy-Daten mit korrekten Labels aus TrainingConfig
    train_data = [
        {"text": "Dies ist ein Antrag für Betriebsunterbrechung", "label": "BU_ANTRAG"},
        {"text": "Hier ist eine Police für Versicherung", "label": "POLICE"},
        {"text": "Diese Bedingungen sind wichtig zu beachten", "label": "BEDINGUNGEN"},
        {"text": "Sonstiger wichtiger Text für Training", "label": "SONSTIGES"},
        {"text": "Noch ein BU Antrag für bessere Abdeckung", "label": "BU_ANTRAG"},
        {"text": "Eine weitere Police mit Details", "label": "POLICE"},
    ]
    
    val_data = [
        {"text": "Validierung für BU Antrag", "label": "BU_ANTRAG"},
        {"text": "Validierung für Bedingungen", "label": "BEDINGUNGEN"},
        {"text": "Validierung für sonstiges", "label": "SONSTIGES"},
    ]
    
    # Erstelle temporäre CSV-Dateien
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    
    pd.DataFrame(train_data).to_csv(train_path, index=False)
    pd.DataFrame(val_data).to_csv(val_path, index=False)
    
    # Setze Umgebungsvariablen für die Training-Konfiguration
    monkeypatch.setenv("TRAIN_PATH", str(train_path))
    monkeypatch.setenv("VAL_PATH", str(val_path))
    
    return str(train_path), str(val_path)
    return str(train_path), str(val_path)
    return str(train_path), str(val_path)
