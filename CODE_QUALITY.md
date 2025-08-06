# Code Quality & Development Guide

## 🛠️ Code-Qualitäts-Tools

Nach dem Upgrade verfügt das BU-Processor-Projekt über professionelle Code-Qualitäts-Tools:

### Verfügbare Tools

1. **Black** - Code-Formatierung
2. **isort** - Import-Sortierung  
3. **flake8** - Linting und Style-Checks
4. **MyPy** - Type-Checking

### Quick Start

```bash
# Development-Dependencies installieren
pip install -r requirements-dev.txt

# Alle Tools ausführen (automatische Reparatur)
python scripts/code_quality.py --fix

# Nur prüfen ohne Änderungen
python scripts/code_quality.py --check

# Nur Type-Checking
python scripts/code_quality.py --mypy-only
```

### Windows Batch Script

```cmd
# Automatische Reparatur
scripts\code_quality.bat fix

# Nur prüfen
scripts\code_quality.bat check

# Nur MyPy
scripts\code_quality.bat mypy-only
```

## 📋 Code-Standards

### Type Hints

Alle öffentlichen Funktionen haben jetzt vollständige Type Hints:

```python
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
```

### Docstring-Standard

Alle Funktionen verwenden Google-Style Docstrings:

```python
def process_document(
    self,
    file_path: Union[str, Path],
    strategy: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> PipelineResult:
    """Hauptmethode: Dokumentenverarbeitung mit Strategy Pattern.
    
    Verarbeitet ein einzelnes Dokument durch die komplette Pipeline mit
    konfigurierbarer Strategie und Custom-Konfiguration.
    
    Args:
        file_path: Pfad zum zu verarbeitenden Dokument
        strategy: Verarbeitungsstrategie ('fast', 'balanced', 'comprehensive')
        custom_config: Optionale Custom-Konfiguration
        
    Returns:
        PipelineResult mit allen Verarbeitungsergebnissen
        
    Raises:
        ValueError: Bei ungültigen Konfigurationen
        ValidationError: Bei Pydantic-Validierungsfehlern
    """
```

## 🔧 Konfiguration

### MyPy (mypy.ini)

```ini
[mypy]
python_version = 3.8
warn_return_any = True
disallow_untyped_defs = False
check_untyped_defs = True
# ... weitere Konfiguration
```

### Black & isort (pyproject.toml)

```toml
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']

[tool.isort]
profile = "black"
line_length = 88
known_first_party = ["bu_processor"]
```

### Flake8 (.flake8)

```ini
[flake8]
max-line-length = 88
ignore = E203, W503, E501
per-file-ignores = __init__.py:F401,F403
```

## 🚀 Entwicklungsworkflow

### Vor jedem Commit

```bash
# 1. Code-Qualität prüfen
python scripts/code_quality.py --check

# 2. Falls Fehler: automatisch reparieren
python scripts/code_quality.py --fix  

# 3. Tests ausführen
pytest tests/

# 4. Committen
git add .
git commit -m "feat: improved function with type hints"
```

### Pre-Commit Hooks

```bash
# Pre-Commit installieren
pip install pre-commit

# Hooks einrichten
pre-commit install

# Manuell ausführen
pre-commit run --all-files
```

## 📊 Code-Qualitäts-Metriken

### Was die Tools prüfen

| Tool | Prüft | Beispiel |
|------|--------|----------|
| **Black** | Code-Formatierung | Einheitliche Klammer-Stile, Zeilenlänge |
| **isort** | Import-Reihenfolge | Stdlib → Third-party → Local |  
| **flake8** | PEP 8 Compliance | Unused imports, naming conventions |
| **MyPy** | Type Safety | Missing type hints, type mismatches |

### Ausgabebeispiel

```
🚀 Code Quality Check gestartet
   Project Root: C:\ml_classifier_poc\bu_processor
   Source Dirs: bu_processor, tests, scripts

🔧 Import sorting with isort...
   ✅ Import sorting with isort - SUCCESS

🔧 Code formatting with black...  
   ✅ Code formatting with black - SUCCESS

🔧 Linting with flake8...
   ✅ Linting with flake8 - SUCCESS

🔧 Type checking with mypy...
   ✅ Type checking with mypy - SUCCESS

🎉 Alle Code-Qualitäts-Checks erfolgreich!
```

## 🔍 Fehlerbehebung

### Häufige MyPy-Fehler

```python
# ❌ Fehler: Missing type hint
def process_data(data):
    return data

# ✅ Korrekt: Mit Type Hints
def process_data(data: List[str]) -> Dict[str, Any]:
    return {"processed": data}
```

### Häufige Flake8-Warnungen

```python
# ❌ F401: 'sys' imported but unused
import sys
import os

def main():
    print("Hello")

# ✅ Korrekt: Nur benötigte Imports
import os

def main():
    print("Hello")
```

### Black vs. Manual Formatting

```python
# ❌ Vor Black
def long_function_name(parameter_one,parameter_two,parameter_three,parameter_four):
    return parameter_one+parameter_two+parameter_three+parameter_four

# ✅ Nach Black
def long_function_name(
    parameter_one, parameter_two, parameter_three, parameter_four
):
    return parameter_one + parameter_two + parameter_three + parameter_four
```

## 📈 Nächste Schritte

1. **CI/CD Integration**: Automatische Code-Qualitäts-Checks in Pipeline
2. **Coverage Reports**: Test-Coverage überwachen  
3. **Documentation**: Automatische API-Dokumentation mit Sphinx
4. **Performance**: Code-Profiling und Performance-Tests

## 🔗 Weitere Ressourcen

- [Black Documentation](https://black.readthedocs.io/)
- [MyPy Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)  
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [Type Hints (PEP 484)](https://peps.python.org/pep-0484/)
