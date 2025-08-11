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


# === PROJEKT-WEITE FIXTURES ===

@pytest.fixture
def sample_pdf_path(test_data_dir):
    """Pfad zur Sample-PDF-Datei."""
    return test_data_dir / "sample.pdf"


@pytest.fixture
def classifier_with_mocks(mocker, mock_tokenizer, mock_torch_model):
    """Vollständig gemockter RealMLClassifier für Tests."""
    # Setup mock components
    mock_tokenizer_instance = mock_tokenizer()
    mock_model_instance = mock_torch_model()
    
    # Mock AutoTokenizer und AutoModel imports
    mocker.patch("bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", 
                 return_value=mock_tokenizer_instance)
    mocker.patch("bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", 
                 return_value=mock_model_instance)
    
    # Mock PyTorch device detection
    mocker.patch("torch.cuda.is_available", return_value=False)
    
    # Mock PDF Extractor
    mock_pdf_extractor = mocker.Mock()
    mock_extracted_content = mocker.Mock()
    mock_extracted_content.text = "Test PDF Inhalt für Klassifikation"
    mock_extracted_content.page_count = 3
    mock_extracted_content.file_path = "test.pdf"
    mock_extracted_content.metadata = {"title": "Test Document"}
    mock_extracted_content.extraction_method = "mocked"
    mock_extracted_content.chunking_enabled = False
    mock_extracted_content.chunking_method = "none"
    
    mock_pdf_extractor.extract_text_from_pdf.return_value = mock_extracted_content
    mocker.patch("bu_processor.pipeline.classifier.EnhancedPDFExtractor", 
                 return_value=mock_pdf_extractor)
    
    # Import und erstelle Classifier
    from bu_processor.pipeline.classifier import RealMLClassifier
    classifier = RealMLClassifier()
    
    return classifier


@pytest.fixture(scope="session")
def project_root():
    """Projekt-Root-Verzeichnis."""
    return Path(__file__).parent.parent


# Frühzeitig sicherstellen, dass das Package importierbar ist
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Test environment flags (used by pinecone_integration, classifier, etc.)
os.environ.setdefault("PYTEST_RUNNING", "1")
os.environ.setdefault("ALLOW_EMPTY_PINECONE_KEY", "1")
os.environ.setdefault("BU_LAZY_MODELS", "1")

# Provide a tiny default model name to avoid large downloads if code checks env
os.environ.setdefault("BUPROC_MODEL_NAME", "sshleifer/tiny-distilroberta-base")

def pytest_configure(config):  # noqa: D401
    """Pytest hook to mark that test config was applied."""
    os.environ["PYTEST_CONFIGURED"] = "1"


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


# === MOCK FACTORIES ===

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
        logits[0, high_confidence_category] = 2.0  # High logit für eine Kategorie
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
        from bu_processor.pipeline.pdf_extractor import DocumentChunk
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size//10):  # ~10 chars per word estimate
            chunk_words = words[i:i+chunk_size//10]
            chunk_text = " ".join(chunk_words)
            
            chunk = DocumentChunk(
                id=f"chunk_{i//10}",
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
    os.environ["OPENAI_API_KEY"] = "test_key_openai"
    os.environ["PINECONE_API_KEY"] = "test_key_pinecone"
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
            mock_outputs.logits = torch.tensor([[0.2, 0.8, 0.0]])
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
