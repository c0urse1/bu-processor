## 📝 Beschreibung
<!-- Beschreibe deine Änderungen kurz und präzise -->

**Was:** <!-- Was wurde geändert? -->

**Warum:** <!-- Warum war diese Änderung notwendig? -->

**Wie:** <!-- Wie wurde das Problem gelöst? -->

## 🔄 Art der Änderung
<!-- Markiere zutreffende Optionen -->
- [ ] 🐛 **Bug fix** (nicht-breaking change, der ein Problem behebt)
- [ ] ✨ **Neue Feature** (nicht-breaking change, der Funktionalität hinzufügt)
- [ ] 💥 **Breaking change** (Fix oder Feature, der bestehende Funktionalität bricht)
- [ ] 📚 **Documentation** (keine Code-Änderungen)
- [ ] 🔧 **Refactoring** (Code-Verbesserungen ohne funktionale Änderungen)
- [ ] ⚡ **Performance** (Leistungsverbesserungen)
- [ ] 🧪 **Tests** (neue oder verbesserte Tests)

## 🧪 Tests
<!-- Beschreibe die Tests für deine Änderungen -->
- [ ] **Unit tests** hinzugefügt/aktualisiert
- [ ] **Integration tests** bestehen
- [ ] **Performance tests** (falls relevant)
- [ ] **Manual testing** durchgeführt

### Test-Kommandos:
```bash
# Tests die ausgeführt wurden
pytest tests/test_xyz.py -v
python cli.py test-scenario
```

## ✅ Checklist
<!-- Stelle sicher, dass alle Punkte erfüllt sind -->

### Code Quality:
- [ ] **Black formatting** angewandt (`black src/ tests/`)
- [ ] **Import sorting** mit isort (`isort src/ tests/`)
- [ ] **Linting** mit flake8 bestanden (`flake8 src/ tests/`)
- [ ] **Type hints** hinzugefügt (`mypy src/`)
- [ ] **Docstrings** für neue/geänderte APIs
- [ ] **Self-review** des Codes durchgeführt

### Funktionalität:
- [ ] **Backwards compatibility** gewährleistet (oder Breaking Change dokumentiert)
- [ ] **Error handling** implementiert
- [ ] **Logging** hinzugefügt (falls relevant)
- [ ] **Configuration** erweitert (falls nötig)

### Dokumentation:
- [ ] **Code-Kommentare** in komplexem Code
- [ ] **README** aktualisiert (falls nötig)
- [ ] **API-Dokumentation** ergänzt
- [ ] **CHANGELOG** aktualisiert (bei Features/Fixes)

### Tests & Performance:
- [ ] **Alle Tests** bestehen lokal
- [ ] **Performance** nicht verschlechtert
- [ ] **Memory usage** berücksichtigt

## 🔗 Verwandte Issues
<!-- Verlinke verwandte Issues -->
Fixes #XXX
Closes #YYY
Related to #ZZZ

## 📊 Änderungs-Impact
<!-- Bewerte den Impact deiner Änderungen -->

**Betroffene Komponenten:**
- [ ] PDF-Extraktion
- [ ] ML-Klassifikation  
- [ ] Semantic Chunking
- [ ] Vector Database (Pinecone)
- [ ] API/Web Interface
- [ ] CLI
- [ ] Configuration System
- [ ] Tests/CI

**Performance Impact:**
- [ ] Keine Auswirkungen
- [ ] Verbesserte Performance
- [ ] Potentielle Performance-Einbußen (dokumentiert)

## 🧪 Test-Szenarien
<!-- Beschreibe, wie deine Änderungen getestet werden können -->

1. **Standard Use Case:**
   ```bash
   python cli.py classify "test text"
   ```

2. **Edge Cases:**
   ```bash
   python cli.py process large_file.pdf
   ```

3. **Integration:**
   ```bash
   python -m pytest tests/integration/
   ```

## 📸 Screenshots
<!-- Falls UI-Änderungen: Before/After Screenshots -->

## 🚀 Deployment Notes
<!-- Spezielle Anweisungen für das Deployment -->
- [ ] **Migration scripts** erforderlich
- [ ] **Environment variables** geändert
- [ ] **Dependencies** aktualisiert
- [ ] **Database changes** (falls relevant)

## 🔍 Review-Fokus
<!-- Worauf sollten Reviewer besonders achten? -->
- Bitte überprüft besonders: **[spezifische Bereiche]**
- Performance-kritische Teile: **[Code-Abschnitte]**
- Neue APIs: **[API-Änderungen]**

## 💭 Offene Fragen
<!-- Fragen oder Unsicherheiten, die du hast -->
- [ ] Ist die API-Design optimal?
- [ ] Sollten weitere Tests hinzugefügt werden?
- [ ] Performance-Optimierungen notwendig?

---

<!-- 
Für Reviewer:
- Prüft den Code auf Lesbarkeit und Wartbarkeit
- Achtet auf Performance-Implications
- Testet kritische Pfade manuell
- Überprüft Dokumentation und Kommentare
-->
