# Test-Fixtures für PDF-Verarbeitung

## Benötigte Test-Dateien

Für die vollständige Funktionalität der PDF-Demo bitte folgende Dateien in diesem Verzeichnis ablegen:

### sample.pdf
Eine Beispiel-PDF-Datei zum Testen der Extraktion.

**Beispiel-Inhalt einer Test-PDF:**
- Ein paar Seiten Text (z.B. aus einem Word-Dokument exportiert)
- Deutsche Inhalte für bessere Klassifikations-Tests
- Verschiedene Schriftarten und Formatierungen

**Erstellen einer Test-PDF:**
1. Word-Dokument mit Beispieltext erstellen
2. Als PDF exportieren
3. Als `sample.pdf` in diesem Verzeichnis speichern

**Beispiel-Textinhalt für Test-PDF:**
```
Berufsfeld: Softwareentwicklung

Ich arbeite seit 5 Jahren als Softwareentwickler bei einem großen Technologieunternehmen. 
Meine Hauptaufgaben umfassen:
- Entwicklung von Web-Anwendungen
- Backend-Systeme mit Python und Java
- Datenbank-Design und -Optimierung
- Agile Projektmethoden

Qualifikationen:
- Bachelor of Science in Informatik
- Erfahrung mit Cloud-Technologien
- DevOps und CI/CD
```

## Test-Kommandos

Nach dem Hinzufügen der sample.pdf:

```bash
# PDF-Extraktion testen
python cli.py pdf tests/fixtures/sample.pdf

# Komplette Demo mit PDF-Support
python cli.py demo

# PDF direkt klassifizieren
python cli.py classify tests/fixtures/sample.pdf
```

## Weitere Test-PDFs

Für umfassendere Tests können mehrere PDFs mit verschiedenen Inhalten hinzugefügt werden:
- `sample_IT.pdf` - IT-Berufe
- `sample_Marketing.pdf` - Marketing-Berufe  
- `sample_Finance.pdf` - Finanz-Berufe
- `corrupted.pdf` - Beschädigte PDF zum Testen der Fehlerbehandlung

## Automatische Test-Generierung

Alternativ kann ein Script zum automatischen Erstellen von Test-PDFs verwendet werden:

```python
# generate_test_pdfs.py
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_test_pdf(filename, content):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, content)
    c.save()

# Test-PDFs erstellen
test_contents = {
    "sample.pdf": "Ich bin Softwareentwickler und arbeite mit Python.",
    "sample_marketing.pdf": "Ich arbeite im Marketing und entwickle Kampagnen.",
    "sample_finance.pdf": "Als Finanzanalyst erstelle ich Berichte und Prognosen."
}

for filename, content in test_contents.items():
    create_test_pdf(filename, content)
```

**Installation für reportlab:**
```bash
pip install reportlab
```
