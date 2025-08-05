# 🤖 BU-Processor: AI-Powered Document Analysis Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Ein hochmoderner, KI-gestützter Pipeline für die automatische Analyse und Klassifikation von deutschen Berufsunfähigkeits-Dokumenten mit semantischem Chunking, Vektor-Datenbank-Integration und intelligenter Deduplication.

## 🚀 Features

### 🧠 **Intelligente Dokumentanalyse**
- **Multi-Engine PDF-Extraktion** mit PyMuPDF, PyPDF2 und Fallback-Mechanismen
- **Semantisches Chunking** mit SentenceTransformers für bessere Segmentierung
- **Hierarchische Dokumentstruktur-Erkennung** (Überschriften, Inhaltsverzeichnisse)
- **Adaptive Chunk-Größen** basierend auf Content-Type

### 🔍 **Machine Learning Pipeline**
- **BERT-basierte Klassifikation** für BU-Dokument-Kategorien
- **Batch-Verarbeitung** für hochperformante Analyse
- **Confidence-basierte Qualitätskontrolle** mit konfigurierbaren Schwellwerten
- **GPU-Acceleration** für schnellere Inferenz

### 🌊 **Erweiterte Datenverarbeitung**
- **SimHash-basierte Deduplication** zur Erkennung semantischer Duplikate
- **Pinecone Vector Database Integration** für Ähnlichkeitssuche
- **Semantic Clustering** mit DBSCAN und Agglomerative Clustering
- **Multi-linguale Embeddings** (DE/EN) mit optimiertem Caching

### 🛠️ **Enterprise-Ready Architecture**
- **Pydantic-basierte Konfiguration** mit Environment-Management
- **Strukturiertes Logging** mit Retry-Mechanismen und Error-Handling
- **RESTful API** mit FastAPI und automatischer Dokumentation
- **Docker-Support** für einfache Deployment

### 🎯 **Spezialisierung für BU-Dokumente**
- **Deutsche Rechtstexte** mit Legal-Text-Optimierung
- **Versicherungsprodukt-Klassifikation** 
- **Compliance-konforme Verarbeitung** sensibler Daten
- **Automatische Qualitätsbewertung** für Dokumentinhalte

## 📋 Anforderungen

- **Python 3.8+**
- **4GB+ RAM** (8GB empfohlen für größere Dokumente)
- **CUDA-kompatible GPU** (optional, für bessere Performance)

## ⚡ Quick Start

### 1. Repository klonen
```bash
git clone https://github.com/yourusername/bu-processor.git
cd bu-processor
```

### 2. Environment einrichten
```bash
# Virtual Environment erstellen
python -m venv venv

# Aktivieren (Windows)
venv\\Scripts\\activate
# Aktivieren (Linux/Mac)
source venv/bin/activate

# Dependencies installieren
pip install -r requirements.txt
```

### 3. Konfiguration
```bash
# Environment-Datei kopieren
cp .env.example .env

# .env nach Bedarf anpassen
# Mindestens ML_MODEL_PATH setzen
```

### 4. Erste Schritte
```python
from src.pipeline.classifier import RealMLClassifier
from src.pipeline.pdf_extractor import EnhancedPDFExtractor, ChunkingStrategy

# Classifier initialisieren
classifier = RealMLClassifier()

# PDF analysieren
result = classifier.classify_pdf(
    \"path/to/document.pdf\",
    chunking_strategy=ChunkingStrategy.SEMANTIC
)

print(f\"Kategorie: {result.category}\")
print(f\"Confidence: {result.confidence:.3f}\")
print(f\"Chunks: {len(result.chunks)}\")
```

## 🏗️ Architektur

```
bu-processor/
├── src/
│   ├── core/
│   │   └── config.py          # Zentrale Konfiguration
│   ├── pipeline/
│   │   ├── pdf_extractor.py   # PDF-Verarbeitung
│   │   ├── classifier.py      # ML-Klassifikation
│   │   ├── semantic_chunking_enhancement.py
│   │   └── simhash_semantic_deduplication.py
│   ├── api/                   # REST API
│   ├── training/              # Model Training
│   └── evaluation/            # Performance Metrics
├── tests/                     # Unit & Integration Tests
├── scripts/                   # Utility Scripts
└── utils/                     # Helper Functions
```

## 🔧 Konfiguration

Die Anwendung nutzt **Pydantic BaseSettings** für typisierte, validierte Konfiguration:

```python
# Environment-Variablen in .env
BU_PROCESSOR_ENVIRONMENT=development
BU_PROCESSOR_ML_MODEL__MODEL_PATH=bert-base-german-cased
BU_PROCESSOR_PDF_PROCESSING__MAX_PDF_SIZE_MB=50
BU_PROCESSOR_VECTOR_DB__ENABLE_VECTOR_DB=false
```

### Verfügbare Environments
- **Development**: Maximale Features, Debug-Logging
- **Staging**: Production-nah mit umfassendem Logging  
- **Production**: Optimiert für Performance und Sicherheit

## 📊 Performance

### Benchmarks (auf Intel i7, 16GB RAM)
- **PDF-Extraktion**: ~2-5 Sekunden pro Dokument (10-50 Seiten)
- **Klassifikation**: ~100-500ms pro Dokument
- **Batch-Verarbeitung**: ~10-50 Dokumente/Minute
- **Semantic Chunking**: ~5-15 Chunks pro Dokument

### Optimierungen
- ✅ **Batch-Verarbeitung** für besseren Durchsatz
- ✅ **Embedding-Caching** reduziert Rechenzeit um 60-80%
- ✅ **GPU-Acceleration** für 3-5x schnellere Inferenz
- ✅ **Parallel PDF-Processing** für große Dokumente

## 🧪 Erweiterte Features

### Semantic Chunking
```python
from src.pipeline.semantic_chunking_enhancement import SemanticClusteringEnhancer

enhancer = SemanticClusteringEnhancer()
result = enhancer.enhance_chunks_with_semantic_clustering(
    chunks=document_chunks,
    content_type=ContentType.LEGAL_TEXT,
    use_hierarchical_context=True
)
```

### Deduplication
```python
from src.pipeline.simhash_semantic_deduplication import SemanticDeduplicator

deduplicator = SemanticDeduplicator()
unique_chunks = deduplicator.deduplicate_chunks_semantic(
    chunks=all_chunks,
    similarity_threshold=0.85
)
```

### Vector Database Integration
```python
# Pinecone Integration aktivieren
BU_PROCESSOR_VECTOR_DB__ENABLE_VECTOR_DB=true
BU_PROCESSOR_VECTOR_DB__PINECONE_API_KEY=your-api-key

# Automatische Ähnlichkeitssuche
similar_docs = classifier.find_similar_documents(
    query_text=\"Berufsunfähigkeitsversicherung\",
    top_k=5
)
```

## 🧪 Testing

```bash
# Unit Tests
python -m pytest tests/ -v

# Coverage Report
python -m pytest tests/ --cov=src --cov-report=html

# Performance Tests
python -m pytest tests/performance/ -v

# Integration Tests
python -m pytest tests/integration/ -v
```

## 📚 API Dokumentation

Starte den API-Server:
```bash
python -m src.api.main
```

Interaktive Dokumentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Beispiel API-Calls
```bash
# Dokument klassifizieren
curl -X POST \"http://localhost:8000/classify/pdf\" \\
  -F \"file=@document.pdf\" \\
  -F \"chunking_strategy=semantic\"

# Batch-Verarbeitung
curl -X POST \"http://localhost:8000/classify/batch\" \\
  -H \"Content-Type: application/json\" \\
  -d '{\"texts\": [\"Text 1\", \"Text 2\"]}'
```

## 🔧 Development Setup

```bash
# Development Dependencies
pip install -r requirements-dev.txt

# Pre-commit Hooks
pre-commit install

# Code Formatting
black src/ tests/
isort src/ tests/

# Type Checking
mypy src/

# Linting
flake8 src/ tests/
```

## 🐳 Docker Deployment

```bash
# Build Image
docker build -t bu-processor .

# Run Container
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env bu-processor

# Docker Compose
docker-compose up -d
```

## 📈 Monitoring & Observability

- **Structured Logging** mit Structlog
- **Prometheus Metrics** auf Port 9100
- **Health Checks** über `/health` endpoint
- **Performance Tracking** für alle Pipeline-Komponenten

## 🤝 Contributing

1. **Fork** das Repository
2. **Feature Branch** erstellen (`git checkout -b feature/amazing-feature`)
3. **Tests** schreiben und ausführen
4. **Commit** mit aussagekräftiger Nachricht
5. **Pull Request** erstellen

### Code Style
- **Black** für Code-Formatierung
- **Type Hints** für alle öffentlichen APIs
- **Docstrings** im Google Style
- **Pytest** für alle Tests

## 📄 License

Dieses Projekt ist unter der **MIT License** lizenziert - siehe [LICENSE](LICENSE) für Details.

## 🙏 Acknowledgments

- **Hugging Face Transformers** für BERT-Modelle
- **SentenceTransformers** für Embedding-Generierung  
- **Scikit-learn** für Clustering-Algorithmen
- **PyMuPDF** für robuste PDF-Verarbeitung
- **Pydantic** für typisierte Konfiguration

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/bu-processor/issues)
- **Dokumentation**: [Wiki](https://github.com/yourusername/bu-processor/wiki)
- **Diskussionen**: [GitHub Discussions](https://github.com/yourusername/bu-processor/discussions)

---

**⭐ Wenn dir dieses Projekt hilft, gib ihm einen Stern auf GitHub!**
