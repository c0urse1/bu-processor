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

### 📦 Installation
```bash
# One-liner Installation
pip install . && cp .env.example .env && echo "✅ Ready to go!"

# Optional: Edit configuration
nano .env  # Linux/Mac
# notepad .env  # Windows
```

### 🚀 Usage (2 Befehle)
```bash
# 1. Einzelnes PDF analysieren
python -m bu_processor.pipeline --input data/sample.pdf --output out/

# 2. Kompletten Ordner verarbeiten
python -m bu_processor.pipeline --input data/ --output out/ --batch
```

### 💻 CLI Power User
```bash
# Semantic Chunking + Klassifikation
python cli.py classify document.pdf --strategy semantic --confidence 0.8

# Batch mit Fortschrittsanzeige
python cli.py batch data/ --output results/ --progress

# Web Interface (Demo & API)
python cli.py web --host 0.0.0.0 --port 8000
# 🌐 Browser: http://localhost:8000

# Interactive Chatbot
python cli.py chat --model gpt-3.5-turbo

# Alle verfügbaren Befehle
python cli.py --help
```

### 🐍 Python API (Minimal)
```python
from bu_processor.pipeline.classifier import RealMLClassifier

# One-Shot Classification
classifier = RealMLClassifier()
result = classifier.classify_pdf("document.pdf")

print(f"📄 {result.category} ({result.confidence:.0%})")
print(f"📊 {len(result.chunks)} chunks extracted")
```

### 🐍 Python API (Erweitert)
```python
from bu_processor.pipeline.classifier import RealMLClassifier
from bu_processor.pipeline.pdf_extractor import ChunkingStrategy, ContentType
from bu_processor.core.config import get_config

# Konfiguration laden
config = get_config()
print(f"🏗️ Environment: {config.environment.value}")

# Classifier mit Custom Settings
classifier = RealMLClassifier(
    confidence_threshold=0.9,
    use_gpu=True
)

# Fortgeschrittene Klassifikation
result = classifier.classify_pdf(
    "complex_document.pdf",
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    content_type=ContentType.LEGAL_TEXT,
    extract_metadata=True
)

# Detaillierte Ergebnisse
print(f"📋 Kategorie: {result.category}")
print(f"🎯 Confidence: {result.confidence:.3f}")
print(f"📄 Seiten: {result.metadata.get('page_count', 'unbekannt')}")
print(f"🧩 Chunks: {len(result.chunks)}")

# Chunk-Details
for i, chunk in enumerate(result.chunks[:3]):
    print(f"  {i+1}. {chunk.content_type.value}: {chunk.text[:100]}...")
```

### 🔧 Development Setup
```bash
# Komplettes Development Environment
git clone https://github.com/yourusername/bu-processor.git
cd bu-processor
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate   # Windows

pip install -r requirements-dev.txt
pre-commit install
cp .env.example .env

# Teste Installation
python cli.py config  # Zeigt aktuelle Konfiguration
python -m pytest tests/ -v  # Laufe Tests
```

### ⚙️ Environment Konfiguration (.env)
```bash
# Basis-Konfiguration (development)
BU_PROCESSOR_ENVIRONMENT=development
BU_PROCESSOR_ML_MODEL__MODEL_PATH=bert-base-german-cased
BU_PROCESSOR_PDF_PROCESSING__MAX_PDF_SIZE_MB=50
BU_PROCESSOR_VECTOR_DB__ENABLE_VECTOR_DB=false

# Optional: Erweiterte Features
# PINECONE_API_KEY=your-pinecone-key  # Vector Database
# OPENAI_API_KEY=your-openai-key      # Chatbot
# BU_PROCESSOR_API__SECRET_KEY=secure-secret-key
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

### 🌍 Environment Management
- **Development**: Maximale Features, Debug-Logging, alle Validierungen
- **Staging**: Production-nah mit umfassendem Logging, Performance-Tests
- **Production**: Optimiert für Performance, Security-hardened, minimales Logging

```bash
# Environment wechseln
export BU_PROCESSOR_ENVIRONMENT=production
# oder in .env: BU_PROCESSOR_ENVIRONMENT=production

# Config überprüfen
python cli.py config
```

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

**Wir freuen uns über Contributions!** Lies unser detailliertes [CONTRIBUTING.md](CONTRIBUTING.md) für alle Details.

### 🚀 Quick Contributor Setup
```bash
# 1. Fork & Clone
git clone https://github.com/YOURUSERNAME/bu-processor.git
cd bu-processor

# 2. Development Environment
pip install -r requirements-dev.txt
pre-commit install

# 3. Feature Branch
git checkout -b feature/amazing-feature

# 4. Code, Test, Commit
black src/ tests/ && pytest tests/ -v
git commit -m "feat: add amazing feature"
git push origin feature/amazing-feature
```

### 📋 Contribution Areas
- 🐛 **Bug Fixes**: [Good First Issues](https://github.com/yourusername/bu-processor/labels/good%20first%20issue)
- ✨ **Features**: API Extensions, Performance, New Extractors
- 📚 **Documentation**: Examples, Guides, API Docs
- 🧪 **Tests**: Coverage, Integration, Performance
- 🔧 **Refactoring**: Code Quality, Architecture

### 📝 Issue & PR Templates
Wir nutzen strukturierte Templates für bessere Zusammenarbeit:
- 🐛 [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md)
- ✨ [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md)
- ❓ [Question/Support](.github/ISSUE_TEMPLATE/question.md)
- 📝 [Pull Request Template](.github/pull_request_template.md)

## 📄 License

Dieses Projekt ist unter der **MIT License** lizenziert - siehe [LICENSE](LICENSE) für Details.

## 🙏 Acknowledgments

- **Hugging Face Transformers** für BERT-Modelle
- **SentenceTransformers** für Embedding-Generierung  
- **Scikit-learn** für Clustering-Algorithmen
- **PyMuPDF** für robuste PDF-Verarbeitung
- **Pydantic** für typisierte Konfiguration

## 📞 Support & Community

### 🆘 Hilfe bekommen
- 🐛 **Bug Reports**: [Create Issue](https://github.com/yourusername/bu-processor/issues/new?template=bug_report.md)
- ✨ **Feature Requests**: [Request Feature](https://github.com/yourusername/bu-processor/issues/new?template=feature_request.md)
- ❓ **Questions**: [Ask Question](https://github.com/yourusername/bu-processor/issues/new?template=question.md)
- 💬 **Diskussionen**: [GitHub Discussions](https://github.com/yourusername/bu-processor/discussions)

### 📚 Ressourcen
- 📖 **Wiki**: [Comprehensive Guides](https://github.com/yourusername/bu-processor/wiki)
- 🎯 **Examples**: [Code Examples](examples/)
- 🔧 **API Docs**: [docs.yourdomain.com](https://docs.yourdomain.com)
- 📊 **Roadmap**: [Project Board](https://github.com/yourusername/bu-processor/projects)

### 🏷️ Labels & Workflow
- `good first issue` - Perfekt für neue Contributors
- `help wanted` - Community Input erwünscht
- `bug` - Bestätigte Bugs
- `enhancement` - Feature Requests
- `documentation` - Docs-Verbesserungen

---

**⭐ Wenn dir dieses Projekt hilft, gib ihm einen Stern auf GitHub!**
