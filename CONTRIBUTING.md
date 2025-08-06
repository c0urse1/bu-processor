# 🤝 Contributing to BU-Processor

**Willkommen bei der BU-Processor Community!** Wir freuen uns über jeden Beitrag - egal ob Bug Fix, neues Feature, Dokumentation oder Verbesserungsvorschläge.

## 📋 Inhaltsverzeichnis

- [🚀 Quick Start für Contributors](#-quick-start-für-contributors)
- [🌿 Branch Policy](#-branch-policy)
- [📝 Issue Guidelines](#-issue-guidelines)
- [🔄 Pull Request Process](#-pull-request-process)
- [🎯 Development Workflow](#-development-workflow)
- [🧪 Testing Guidelines](#-testing-guidelines)
- [📚 Documentation Standards](#-documentation-standards)
- [🔧 Code Quality Standards](#-code-quality-standards)

## 🚀 Quick Start für Contributors

### 1. **Setup Development Environment**
```bash
# Fork das Repository auf GitHub
# Dann:
git clone https://github.com/YOURUSERNAME/bu-processor.git
cd bu-processor

# Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Development Dependencies
pip install -r requirements-dev.txt

# Pre-commit Hooks (WICHTIG!)
pre-commit install

# Environment Setup
cp .env.example .env
nano .env  # Konfiguration anpassen

# Teste Setup
python cli.py config
python -m pytest tests/ -v --tb=short
```

### 2. **Erstes Contribution**
```bash
# Feature Branch erstellen
git checkout -b feature/your-amazing-feature

# Code changes...
# Tests schreiben...

# Code Quality Checks (automatisch via pre-commit)
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/

# Tests laufen lassen
python -m pytest tests/ -v --cov=src

# Commit und Push
git add .
git commit -m "feat: add amazing feature with tests"
git push origin feature/your-amazing-feature

# Pull Request auf GitHub erstellen
```

## 🌿 Branch Policy

### **Branch Struktur**
```
main           # 🏠 Production-ready stable code
├── develop    # 🔄 Integration branch for features
├── feature/*  # ✨ New features (feature/semantic-chunking)
├── bugfix/*   # 🐛 Bug fixes (bugfix/pdf-extraction-error)
├── hotfix/*   # 🚨 Critical production fixes
└── docs/*     # 📚 Documentation updates
```

### **Branch Naming Convention**
```bash
# ✨ Features (new functionality)
feature/semantic-deduplication
feature/api-authentication
feature/german-language-support

# 🐛 Bug Fixes
bugfix/pdf-memory-leak
bugfix/classification-accuracy
bugfix/config-validation

# 🚨 Hotfixes (critical production issues)
hotfix/security-vulnerability
hotfix/data-corruption

# 📚 Documentation
docs/api-examples
docs/installation-guide
docs/performance-tuning

# 🔧 Refactoring
refactor/pipeline-architecture
refactor/config-system

# 🧪 Experimental
experiment/new-ml-model
experiment/alternative-chunking
```

### **Branch Protection Rules**
- **main**: Requires PR review + status checks + up-to-date
- **develop**: Requires PR review + status checks
- **feature/\***: No restrictions, aber pre-commit hooks required

### **Merge Strategy**
- **main ← develop**: `Squash and merge` (clean history)
- **develop ← feature**: `Merge commit` (feature context preserved)
- **hotfix → main**: `Squash and merge` + immediate cherry-pick to develop

## 📝 Issue Guidelines

### **Issue Types & Templates**

#### 🐛 **Bug Reports**
**Template**: [`.github/ISSUE_TEMPLATE/bug_report.md`]
```markdown
---
name: 🐛 Bug Report
about: Report a bug to help us improve
labels: bug, needs-triage
---

## 🐛 Bug Description
Clear description of the bug.

## 🔄 Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

## 🎯 Expected Behavior
What you expected to happen.

## 💥 Actual Behavior
What actually happened.

## 🖥️ Environment
- OS: [e.g. Windows 11, Ubuntu 22.04]
- Python Version: [e.g. 3.9.7]
- BU-Processor Version: [e.g. 3.0.0]
- GPU: [e.g. NVIDIA RTX 3080, None]

## 📋 Additional Context
- Configuration (.env settings)
- Log output
- Screenshots
- Sample files (if applicable)
```

#### ✨ **Feature Requests**
**Template**: [`.github/ISSUE_TEMPLATE/feature_request.md`]
```markdown
---
name: ✨ Feature Request
about: Suggest a new feature
labels: enhancement, needs-triage
---

## 🎯 Feature Summary
Brief description of the feature.

## 💡 Motivation
Why is this feature needed? What problem does it solve?

## 📋 Detailed Description
Detailed explanation of the feature.

## 🎨 Implementation Ideas
Any ideas on how this could be implemented?

## 📊 Additional Context
- Use cases
- Alternative solutions considered
- Screenshots/mockups
```

#### ❓ **Questions & Support**
**Template**: [`.github/ISSUE_TEMPLATE/question.md`]
```markdown
---
name: ❓ Question
about: Ask for help or clarification
labels: question, support
---

## ❓ Question
What do you need help with?

## 🎯 Context
- What are you trying to achieve?
- What have you already tried?
- Any error messages?

## 🖥️ Environment
- OS & Python Version
- BU-Processor Version
- Configuration details
```

### **Issue Labels**

#### **Type Labels**
- `bug` - Something isn't working
- `enhancement` - New feature or request
- `question` - Further information is requested
- `documentation` - Improvements to docs
- `performance` - Performance related
- `security` - Security related

#### **Priority Labels**
- `critical` - Breaks functionality, immediate attention
- `high` - Important, should be next release
- `medium` - Normal priority
- `low` - Nice to have

#### **Status Labels**
- `needs-triage` - Needs initial review
- `needs-info` - Waiting for more information
- `help-wanted` - Community input welcome
- `good-first-issue` - Good for new contributors
- `wontfix` - Not going to be fixed
- `duplicate` - Duplicate issue

#### **Component Labels**
- `api` - REST API related
- `ml-model` - Machine Learning components
- `pdf-extraction` - PDF processing
- `config` - Configuration system
- `cli` - Command line interface
- `web-ui` - Web interface
- `tests` - Testing related

## 🔄 Pull Request Process

### **PR Template**
**Template**: [`.github/pull_request_template.md`]
```markdown
## 📋 PR Summary
Brief description of changes.

## 🎯 Related Issues
Closes #123
Relates to #456

## 🔄 Type of Change
- [ ] 🐛 Bug fix (non-breaking change)
- [ ] ✨ New feature (non-breaking change)
- [ ] 💥 Breaking change
- [ ] 📚 Documentation update
- [ ] 🔧 Refactoring
- [ ] 🧪 Tests only

## 🧪 Testing
- [ ] Unit tests pass locally
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## 📝 Changes Made
### Added
- New semantic chunking algorithm
- API endpoint for batch processing

### Changed
- Improved PDF extraction performance
- Updated configuration validation

### Removed
- Deprecated legacy extraction method

## 🔍 Review Checklist
- [ ] Code follows project standards (black, isort, flake8, mypy)
- [ ] Tests added/updated for new functionality
- [ ] Documentation updated (if needed)
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed

## 📸 Screenshots (if applicable)
Add screenshots for UI changes.

## 📝 Additional Notes
Any additional context or notes for reviewers.
```

### **PR Review Process**

#### **1. Automated Checks** (Must pass before review)
- ✅ **Pre-commit hooks**: Black, isort, flake8, mypy
- ✅ **Tests**: All unit and integration tests
- ✅ **Coverage**: Minimum 80% test coverage
- ✅ **Security**: No secrets or sensitive data
- ✅ **Dependencies**: No unauthorized new dependencies

#### **2. Code Review** (Human review)
- 👀 **Functionality**: Does it work as intended?
- 🏗️ **Architecture**: Fits well with existing codebase?
- 🎯 **Performance**: No significant performance regression?
- 🔒 **Security**: No security vulnerabilities?
- 📚 **Documentation**: Changes are documented?
- 🧪 **Tests**: Adequate test coverage?

#### **3. Approval & Merge**
- **Required**: 1 approving review from maintainer
- **Optional**: Additional reviews for complex changes
- **Merge**: Squash and merge (clean history)

### **PR Best Practices**

#### ✅ **DO**
- Keep PRs small and focused (< 500 lines when possible)
- Write descriptive commit messages
- Add tests for new functionality
- Update documentation
- Reference related issues
- Respond to review feedback promptly

#### ❌ **DON'T**
- Mix multiple unrelated changes
- Skip testing your changes
- Commit directly to main/develop
- Include personal configuration files
- Add large binary files without approval

## 🎯 Development Workflow

### **Standard Feature Development**
```bash
# 1. Sync with latest
git checkout develop
git pull upstream develop

# 2. Create feature branch
git checkout -b feature/semantic-enhancement

# 3. Develop with TDD
# - Write test first
# - Implement feature
# - Refactor if needed
# - Commit atomically

# 4. Regular commits
git add src/new_feature.py tests/test_new_feature.py
git commit -m "feat: add semantic enhancement with confidence scoring"

# 5. Pre-merge preparation
git rebase develop  # Clean up history
python -m pytest tests/ -v --cov=src
pre-commit run --all-files

# 6. Push and create PR
git push origin feature/semantic-enhancement
# Create PR via GitHub UI
```

### **Hotfix Workflow**
```bash
# 1. Emergency branch from main
git checkout main
git checkout -b hotfix/critical-security-fix

# 2. Minimal fix
# - Fix only the critical issue
# - Add regression test
# - Update version

# 3. Test thoroughly
python -m pytest tests/ -v
python cli.py --version

# 4. Dual merge
# PR to main (immediate release)
# Cherry-pick to develop
```

## 🧪 Testing Guidelines

### **Testing Philosophy**
- **Test-Driven Development**: Write tests first when possible
- **Comprehensive Coverage**: Aim for 85%+ test coverage
- **Fast Feedback**: Unit tests should run in < 30 seconds
- **Real-world Testing**: Integration tests with actual PDFs

### **Test Structure**
```
tests/
├── unit/           # Fast isolated tests
├── integration/    # Component interaction tests
├── performance/    # Performance regression tests
├── fixtures/       # Test data (sample PDFs, configs)
└── conftest.py     # Shared pytest configuration
```

### **Testing Commands**
```bash
# All tests
python -m pytest tests/ -v

# Unit tests only (fast)
python -m pytest tests/unit/ -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Performance tests
python -m pytest tests/performance/ -v --benchmark-only

# Specific test
python -m pytest tests/test_classifier.py::test_pdf_classification -v

# Test discovery
python -m pytest --collect-only
```

### **Test Writing Standards**
```python
import pytest
from unittest.mock import Mock, patch
from bu_processor.pipeline.classifier import RealMLClassifier

class TestRealMLClassifier:
    """Test class for RealMLClassifier functionality."""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance for testing."""
        return RealMLClassifier(confidence_threshold=0.8)
    
    def test_pdf_classification_success(self, classifier, sample_pdf):
        """Test successful PDF classification."""
        # Given
        pdf_path = sample_pdf("legal_document.pdf")
        
        # When
        result = classifier.classify_pdf(pdf_path)
        
        # Then
        assert result.category is not None
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.chunks) > 0
    
    @patch('bu_processor.pipeline.classifier.torch.cuda.is_available')
    def test_gpu_fallback_cpu(self, mock_cuda, classifier):
        """Test graceful fallback to CPU when GPU unavailable."""
        # Given
        mock_cuda.return_value = False
        
        # When
        result = classifier.classify_pdf("test.pdf")
        
        # Then
        assert result is not None
        # GPU should gracefully fallback to CPU
```

## 📚 Documentation Standards

### **Docstring Style** (Google Format)
```python
def classify_pdf(
    self,
    pdf_path: str,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
    confidence_threshold: Optional[float] = None
) -> ClassificationResult:
    """Klassifiziert ein PDF-Dokument in BU-relevante Kategorien.
    
    Verwendet semantisches Chunking und BERT-basierte Klassifikation
    für deutsche Berufsunfähigkeits-Dokumente.
    
    Args:
        pdf_path: Pfad zur PDF-Datei (absolut oder relativ)
        chunking_strategy: Strategie für Text-Segmentierung
        confidence_threshold: Schwellwert für sichere Klassifikation
            Falls None, wird Klassenwert verwendet.
    
    Returns:
        ClassificationResult mit Kategorie, Confidence-Score,
        extrahierten Chunks und optionalen Metadaten.
    
    Raises:
        FileNotFoundError: PDF-Datei nicht gefunden
        ValueError: Ungültiger confidence_threshold (nicht 0-1)
        PDFExtractionError: PDF konnte nicht verarbeitet werden
        ClassificationError: ML-Model-Fehler
    
    Example:
        >>> classifier = RealMLClassifier()
        >>> result = classifier.classify_pdf("bu_document.pdf")
        >>> print(f"Kategorie: {result.category}")
        >>> print(f"Sicherheit: {result.confidence:.2%}")
    
    Note:
        Große PDFs (>50MB) werden automatisch in Chunks verarbeitet.
        GPU-Acceleration wird verwendet falls verfügbar.
    """
```

### **README Sections**
- **Klare Feature-Beschreibung** mit Emojis
- **Quick Start** mit Copy-Paste-Commands
- **Architektur-Diagramm** 
- **API-Beispiele** mit Real-World Use Cases
- **Performance-Benchmarks**
- **Troubleshooting** Section

### **Changelog Maintenance**
```markdown
# Changelog

## [3.1.0] - 2025-02-15
### Added
- 🎯 Semantic clustering for improved chunk organization
- 🔍 Enhanced deduplication with SimHash algorithm
- 🌐 Web interface for interactive document analysis

### Changed
- ⚡ Improved PDF extraction performance by 40%
- 🏗️ Refactored configuration system to use Pydantic

### Fixed
- 🐛 Memory leak in batch processing
- 🔧 GPU detection on Windows systems

### Deprecated
- 📉 Legacy extraction method (will be removed in v4.0)
```

## 🔧 Code Quality Standards

### **Pre-Commit Hooks** (automatisch)
```yaml
# .pre-commit-config.yaml
repos:
- repo: https://github.com/psf/black
  rev: 22.3.0
  hooks:
  - id: black
    language_version: python3.9

- repo: https://github.com/pycqa/isort
  rev: 5.10.1
  hooks:
  - id: isort

- repo: https://github.com/pycqa/flake8
  rev: 4.0.1
  hooks:
  - id: flake8
    additional_dependencies: [flake8-docstrings]

- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v0.942
  hooks:
  - id: mypy
    additional_dependencies: [types-requests, types-PyYAML]
```

### **Code Style Enforcement**
```bash
# Automatisch via pre-commit, manuell:
black src/ tests/                    # Code formatting
isort src/ tests/ --profile black    # Import sorting
flake8 src/ tests/                   # Linting
mypy src/                           # Type checking

# Alle auf einmal
python scripts/code_quality.py
# oder
pre-commit run --all-files
```

### **Code Standards**

#### **Python Code Quality**
- ✅ **Type Hints**: Für alle öffentlichen APIs obligatorisch
- ✅ **Docstrings**: Google-Style für alle public functions/classes
- ✅ **Error Handling**: Spezifische Exceptions mit hilfreichen Messages
- ✅ **Logging**: Strukturiertes Logging statt print()
- ✅ **Configuration**: Pydantic BaseSettings statt globale Variablen
- ✅ **Testing**: Minimum 80% Coverage, aber Qualität > Quantität

#### **Security Guidelines**
- 🔒 **No Secrets in Code**: Alle API Keys über Environment Variables
- 🔒 **Input Validation**: Validiere alle externen Inputs
- 🔒 **File Handling**: Sichere PDF-Verarbeitung, Size Limits
- 🔒 **Dependencies**: Regelmäßige Security-Updates

#### **Performance Guidelines**
- ⚡ **Caching**: Cache teure Operationen (Embeddings, Model Loading)
- ⚡ **Batch Processing**: Für ML-Operationen
- ⚡ **Memory Management**: Große PDFs streaming verarbeiten
- ⚡ **Async Where Appropriate**: I/O-bound operations

## 🎯 Contribution Areas

### 🥇 **High Priority**
- 🐛 **Bug Fixes**: Memory leaks, accuracy issues
- ⚡ **Performance**: Faster PDF processing, GPU optimization
- 🔒 **Security**: Input validation, dependency updates
- 📊 **Monitoring**: Better metrics, error tracking

### 🥈 **Medium Priority**
- ✨ **New Features**: Additional ML models, export formats
- 🌐 **API Extensions**: New endpoints, better error handling
- 📚 **Documentation**: Tutorials, examples, guides
- 🧪 **Testing**: Edge cases, performance tests

### 🥉 **Good First Issues**
- 📝 **Documentation**: Fix typos, add examples
- 🧹 **Code Cleanup**: Remove dead code, improve comments
- 🎨 **UI Improvements**: Web interface enhancements
- 📋 **Configuration**: Add new config options

### 🔍 **Current Focus Areas**
1. **German Language Optimization**: Bessere Sprachmodelle
2. **BU-Specific Features**: Spezialisierte Klassifikatoren
3. **Production Readiness**: Monitoring, Error Handling
4. **Documentation**: User Guides, API Examples

## 📊 Definition of Done

### **Feature Development**
- ✅ Code implemented and tested locally
- ✅ Unit tests with 80%+ coverage
- ✅ Integration tests for user scenarios
- ✅ Documentation updated (README, docstrings)
- ✅ Performance impact assessed
- ✅ Security review completed
- ✅ Pre-commit hooks pass
- ✅ PR review approved
- ✅ CI/CD pipeline green

### **Bug Fixes**
- ✅ Root cause identified
- ✅ Fix implemented with minimal scope
- ✅ Regression test added
- ✅ Documentation updated (if needed)
- ✅ Manual verification completed
- ✅ No performance regression

## 🆘 Getting Help

### **Community Support**
- 💬 **GitHub Discussions**: [github.com/yourusername/bu-processor/discussions](https://github.com/yourusername/bu-processor/discussions)
- 🐛 **Bug Reports**: [Create Issue](https://github.com/yourusername/bu-processor/issues/new?template=bug_report.md)
- ❓ **Questions**: [Ask Question](https://github.com/yourusername/bu-processor/issues/new?template=question.md)

### **Maintainer Contact**
- **Lead Maintainer**: [@yourusername](https://github.com/yourusername)
- **Response Time**: Usually within 24-48 hours
- **Complex Issues**: May take longer, aber we'll keep you updated

### **Development Resources**
- 📖 **Technical Documentation**: [Wiki](https://github.com/yourusername/bu-processor/wiki)
- 🎯 **Architecture Guide**: [docs/architecture.md](docs/architecture.md)
- 🧪 **Testing Guide**: [docs/testing.md](docs/testing.md)
- 🚀 **Deployment Guide**: [docs/deployment.md](docs/deployment.md)

## 🙏 Recognition

### **Contributors Hall of Fame**
All contributors werden im [CONTRIBUTORS.md](CONTRIBUTORS.md) gelistet.

### **Contribution Types**
Wir erkennen alle Arten von Beiträgen:
- 💻 **Code**: Features, Fixes, Refactoring
- 📚 **Documentation**: Guides, Examples, API Docs
- 🐛 **Bug Reports**: Detailed, reproducible issues
- 💡 **Ideas**: Feature suggestions, improvements
- 🎨 **Design**: UI/UX improvements
- 📊 **Data**: Training data, benchmarks
- 🔍 **Review**: Code review, testing
- 📢 **Advocacy**: Spreading the word, tutorials

### **Monthly Contributor Spotlight**
Jeden Monat heben wir besonders wertvolle Contributions hervor!

---

## 📜 Code of Conduct

Wir folgen dem [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). 

**TL;DR**: Sei respektvoll, hilfsbereit und konstruktiv. Wir sind eine freundliche Community! 🌟

---

**🎉 Danke, dass du zu BU-Processor beitragen möchtest! Gemeinsam bauen wir die beste Lösung für deutsche BU-Dokument-Analyse!**
