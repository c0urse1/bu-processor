# 🤝 Contributing to BU-Processor

Vielen Dank für dein Interesse an der Weiterentwicklung des BU-Processor! Diese Anleitung hilft dir dabei, erfolgreich zum Projekt beizutragen.

## 🚀 Quick Start für Contributors

### 1. Repository Setup
```bash
# Repository forken und klonen
git clone https://github.com/YOURUSERNAME/bu-processor.git
cd bu-processor

# Development Environment einrichten
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\\Scripts\\activate  # Windows

# Alle Dependencies installieren
pip install -r requirements-dev.txt

# Pre-commit hooks installieren
pre-commit install
```

### 2. Branch erstellen
```bash
# Aktuellen main branch pullen
git checkout main
git pull origin main

# Feature branch erstellen
git checkout -b feature/deine-neue-feature
# oder für Bugfixes: git checkout -b fix/bugfix-beschreibung
```

## 📋 Development Guidelines

### Code Style & Quality

**Wir verwenden:**
- **Black** für automatische Code-Formatierung
- **isort** für Import-Sortierung
- **flake8** für Linting
- **mypy** für Type-Checking
- **pytest** für Tests

**Vor jedem Commit ausführen:**
```bash
# Code formatieren
black src/ tests/
isort src/ tests/

# Linting prüfen
flake8 src/ tests/

# Type-Checking
mypy src/

# Tests ausführen
pytest tests/ -v
```

### Code Conventions

#### 1. **Type Hints überall**
```python
def process_document(text: str, max_length: Optional[int] = None) -> DocumentResult:
    \"\"\"Process document with optional length limit.\"\"\"
    pass
```

#### 2. **Aussagekräftige Funktionsnamen**
```python
# ✅ Gut
def extract_text_from_pdf_with_fallback(pdf_path: Path) -> ExtractedContent:
    pass

# ❌ Schlecht  
def process_pdf(path):
    pass
```

#### 3. **Docstrings im Google Style**
```python
def enhance_chunks_with_semantic_clustering(
    chunks: List[ChunkProtocol], 
    content_type: Optional[ContentType] = None
) -> SemanticClusterResult:
    \"\"\"Erweitert Chunks um semantische Clustering-Information.
    
    Args:
        chunks: Liste der zu analysierenden Chunks
        content_type: Content-Typ für adaptive Clustering-Parameter
        
    Returns:
        SemanticClusterResult mit erweiterten Chunks und Analysedaten
        
    Raises:
        ValueError: Wenn keine Chunks bereitgestellt werden
        RuntimeError: Wenn Embedding-Modell nicht verfügbar ist
    \"\"\"
    pass
```

#### 4. **Strukturiertes Logging**
```python
import structlog

logger = structlog.get_logger(\"module.name\")

# ✅ Gut
logger.info(\"PDF processing started\", 
           file_path=str(pdf_path),
           pages=page_count,
           strategy=chunking_strategy.value)

# ❌ Schlecht
logger.info(f\"Processing {pdf_path} with {page_count} pages\")
```

#### 5. **Error Handling mit Context**
```python
# ✅ Gut
try:
    result = process_document(text)
except ValidationError as e:
    logger.error(f\"Document validation failed: {e}\", exc_info=True)
    raise DocumentProcessingError(f\"Invalid document format: {e}\") from e

# ❌ Schlecht
try:
    result = process_document(text)
except Exception as e:
    print(f\"Error: {e}\")
```

## 🧪 Testing Guidelines

### Test Structure
```bash
tests/
├── unit/              # Unit tests für einzelne Funktionen
├── integration/       # Integration tests für Komponenten
├── performance/       # Performance & Benchmark tests
├── fixtures/          # Test-Daten und Fixtures
└── conftest.py       # Pytest Konfiguration
```

### Test Conventions

#### 1. **Aussagekräftige Testnamen**
```python
def test_pdf_extractor_handles_corrupted_files_gracefully():
    \"\"\"Test that PDF extractor fails gracefully with corrupted files.\"\"\"
    pass

def test_semantic_clustering_creates_expected_number_of_clusters():
    \"\"\"Test that clustering produces reasonable cluster count.\"\"\"
    pass
```

#### 2. **Arrange-Act-Assert Pattern**
```python
def test_chunk_classification_with_high_confidence():
    # Arrange
    classifier = RealMLClassifier()
    test_text = \"Berufsunfähigkeitsversicherung mit hoher Qualität\"
    
    # Act
    result = classifier.classify_text(test_text)
    
    # Assert
    assert result.confidence > 0.8
    assert result.is_confident is True
    assert isinstance(result.category, int)
```

#### 3. **Fixtures für Setup**
```python
@pytest.fixture
def sample_classifier():
    \"\"\"Provides a configured classifier for testing.\"\"\"
    return RealMLClassifier(
        model_path=\"bert-base-german-cased\",
        batch_size=8,
        max_retries=1
    )
```

### Test Coverage
- **Minimum 80% Coverage** für neue Features
- **100% Coverage** für kritische Pipeline-Komponenten
- **Integration Tests** für alle API-Endpoints

```bash
# Coverage Report generieren
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

## 📖 Documentation

### Inline Documentation
- **Alle öffentlichen APIs** müssen Docstrings haben
- **Komplexe Algorithmen** brauchen Inline-Kommentare
- **Configuration Options** müssen dokumentiert sein

### README Updates
- **Neue Features** in README.md erwähnen
- **Breaking Changes** prominent markieren
- **Beispiele** für neue APIs hinzufügen

## 🔍 Pull Request Process

### 1. **PR-Checklist**
- [ ] **Tests** für neue Features geschrieben
- [ ] **Alle Tests** bestehen lokal
- [ ] **Code formatiert** mit black/isort
- [ ] **Type hints** hinzugefügt
- [ ] **Docstrings** für neue APIs
- [ ] **README** aktualisiert (falls nötig)
- [ ] **Breaking changes** dokumentiert

### 2. **PR-Template**
```markdown
## Beschreibung
Kurze Beschreibung der Änderungen.

## Art der Änderung
- [ ] Bug fix (nicht-breaking change)
- [ ] Neue Feature (nicht-breaking change)
- [ ] Breaking change (fix oder feature)
- [ ] Documentation update

## Tests
- [ ] Unit tests hinzugefügt/aktualisiert
- [ ] Integration tests bestehen
- [ ] Performance tests (falls relevant)

## Checklist
- [ ] Code folgt Projekt-Style-Guidelines
- [ ] Self-review durchgeführt
- [ ] Kommentare in schwer verständlichem Code
- [ ] Dokumentation aktualisiert
```

### 3. **Review Process**
1. **Automatische Checks** müssen bestehen (CI/CD)
2. **Code Review** von mindestens einem Maintainer
3. **Testing** in verschiedenen Environments
4. **Documentation Review** falls nötig

## 🐛 Bug Reports

### Bug Report Template
```markdown
**Beschreibung**
Klare Beschreibung des Bugs.

**Reproduktion**
Schritte zur Reproduktion:
1. Gehe zu '...'
2. Klicke auf '...'
3. Scrolle nach unten zu '...'
4. Sieh Fehler

**Erwartetes Verhalten**
Was sollte passieren.

**Screenshots**
Falls anwendbar, Screenshots hinzufügen.

**Environment:**
 - OS: [e.g. Windows 10]
 - Python Version: [e.g. 3.9.0]
 - BU-Processor Version: [e.g. 1.0.0]

**Zusätzlicher Kontext**
Weitere Details zum Problem.
```

## 💡 Feature Requests

### Feature Request Template
```markdown
**Ist dein Feature-Request mit einem Problem verbunden?**
Klare Beschreibung des Problems. z.B. \"Ich bin frustriert wenn [...]\"

**Beschreibe die gewünschte Lösung**
Klare Beschreibung was du möchtest.

**Beschreibe Alternativen**
Andere Lösungsansätze die du betrachtet hast.

**Zusätzlicher Kontext**
Screenshots, Mockups, etc.
```

## 🏷️ Version & Release Process

### Semantic Versioning
Wir folgen [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: Neue Features (backward compatible)  
- **PATCH**: Bug fixes (backward compatible)

### Release Workflow
1. **Feature Development** in Feature-Branches
2. **Integration** in `develop` branch
3. **Release Candidates** für Testing
4. **Release** in `main` branch mit Git Tags

## 🆘 Hilfe & Support

### Wo du Hilfe bekommst:
- **GitHub Issues** für Bugs und Feature Requests
- **GitHub Discussions** für allgemeine Fragen
- **Code Comments** für Implementation-Details
- **Tests** als Dokumentation der erwarteten API

### Kommunikation
- **Deutsch oder Englisch** in Issues/PRs
- **Höflich und konstruktiv** in Reviews
- **Konkrete Beispiele** bei Problemen/Vorschlägen

## 🎯 Areas für Contributions

### High Impact Areas:
1. **Performance Optimierung** der ML-Pipeline
2. **Neue PDF-Extraction** Engines hinzufügen
3. **API-Endpoints** erweitern
4. **Test Coverage** verbessern
5. **Documentation** vervollständigen

### Good First Issues:
- Bug fixes in Error Handling
- Zusätzliche Logging-Messages
- Test Cases für Edge Cases
- README-Verbesserungen
- Code-Kommentare hinzufügen

---

**Vielen Dank für deinen Beitrag zum BU-Processor! 🚀**

Bei Fragen kannst du gerne ein Issue öffnen oder in den Discussions nachfragen.
