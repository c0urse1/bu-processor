#!/usr/bin/env python3
"""
Test-PDF-Generator für BU-Classifier Demo
=========================================
Erstellt Beispiel-PDFs mit verschiedenen Berufsinhalten für Tests
"""

import os
from pathlib import Path
from typing import Dict, List

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Test-Inhalte für verschiedene Berufsfelder
TEST_CONTENTS = {
    "sample.pdf": {
        "title": "Software-Entwickler Profil",
        "content": [
            "BERUFSFELD: SOFTWARE-ENTWICKLUNG",
            "",
            "Ich arbeite seit 5 Jahren als Softwareentwickler bei einem",
            "mittelständischen Technologieunternehmen in München.",
            "",
            "TÄTIGKEITEN:",
            "• Entwicklung von Web-Anwendungen mit Python/Django",
            "• Backend-Services und REST-APIs",
            "• Datenbank-Design mit PostgreSQL",
            "• Code-Reviews und Team-Kollaboration",
            "• Agile Entwicklung in Scrum-Teams",
            "",
            "QUALIFIKATIONEN:",
            "• Bachelor Informatik (TU München)",
            "• Python, JavaScript, SQL, Docker",
            "• AWS Cloud-Zertifizierung",
            "• 5+ Jahre Berufserfahrung",
            "",
            "AKTUELLE PROJEKTE:",
            "Derzeit entwickle ich eine E-Commerce-Plattform",
            "mit Microservices-Architektur und implementiere",
            "CI/CD-Pipelines für automatische Deployments."
        ]
    },
    
    "sample_marketing.pdf": {
        "title": "Marketing Manager Profil", 
        "content": [
            "BERUFSFELD: DIGITAL MARKETING",
            "",
            "Als Marketing Manager bin ich verantwortlich für die",
            "Entwicklung und Umsetzung von Marketing-Strategien.",
            "",
            "AUFGABEN:",
            "• Kampagnen-Entwicklung für Social Media",
            "• Content-Marketing und SEO-Optimierung",
            "• Budget-Planung und ROI-Analyse",
            "• Marktforschung und Zielgruppen-Analyse",
            "• Zusammenarbeit mit externen Agenturen",
            "",
            "TOOLS & SKILLS:",
            "• Google Analytics, AdWords, Facebook Ads",
            "• Adobe Creative Suite, Canva",
            "• Marketing Automation (HubSpot, Mailchimp)",
            "• A/B-Testing und Conversion-Optimierung",
            "",
            "ERFOLGE:",
            "Steigerung der Online-Conversions um 45% durch",
            "optimierte Landing Pages und zielgruppenspezifische",
            "Kampagnen im letzten Quartal."
        ]
    },
    
    "sample_finance.pdf": {
        "title": "Finanzanalyst Profil",
        "content": [
            "BERUFSFELD: FINANCIAL SERVICES",
            "",
            "Ich arbeite als Senior Financial Analyst bei einer",
            "internationalen Unternehmensberatung.",
            "",
            "KERNKOMPETENZEN:",
            "• Finanzmodellierung und Bewertungsanalysen",
            "• Due Diligence bei M&A-Transaktionen",
            "• Risikobewertung und Portfolio-Management",
            "• Quartalsberichterstattung und Forecasting",
            "• Investor Relations und Präsentationen",
            "",
            "SOFTWARE:",
            "• Excel (VBA, Pivot-Tabellen, Makros)",
            "• Bloomberg Terminal, Reuters Eikon",
            "• SAP, Oracle Financials",
            "• Python für Datenanalyse",
            "",
            "AUSBILDUNG:",
            "• Master in Finance (Frankfurt School)",
            "• CFA Level II Kandidat",
            "• 7+ Jahre Berufserfahrung",
            "",
            "AKTUELLE MANDATE:",
            "Leitung der Finanzanalyse für eine €500M",
            "Akquisition im Technologiesektor."
        ]
    }
}

def create_pdf_with_reportlab(filename: str, title: str, content: List[str], output_dir: Path):
    """Erstelle PDF mit reportlab (falls verfügbar)"""
    filepath = output_dir / filename
    
    # Canvas erstellen
    c = canvas.Canvas(str(filepath), pagesize=A4)
    width, height = A4
    
    # Titel
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, title)
    
    # Content
    c.setFont("Helvetica", 11)
    y_position = height - 100
    
    for line in content:
        if y_position < 50:  # Neue Seite
            c.showPage()
            c.setFont("Helvetica", 11)
            y_position = height - 50
            
        if line.startswith("•"):
            c.drawString(70, y_position, line)
        elif line.isupper() or line.endswith(":"):
            c.setFont("Helvetica-Bold", 11)
            c.drawString(50, y_position, line)
            c.setFont("Helvetica", 11)
        else:
            c.drawString(50, y_position, line)
            
        y_position -= 15
    
    c.save()
    return filepath

def create_pdf_fallback(filename: str, title: str, content: List[str], output_dir: Path):
    """Fallback: Erstelle einfache Text-Datei als Ersatz"""
    txt_filename = filename.replace('.pdf', '.txt')
    filepath = output_dir / txt_filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        for line in content:
            f.write(f"{line}\n")
    
    return filepath

def generate_test_pdfs(output_dir: str = "tests/fixtures"):
    """Hauptfunktion zum Generieren der Test-PDFs"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("📄 Test-PDF-Generator")
    print("====================")
    
    if not REPORTLAB_AVAILABLE:
        print("⚠️  reportlab nicht installiert - verwende Text-Dateien als Fallback")
        print("   Installation: pip install reportlab")
        print()
    
    created_files = []
    
    for filename, data in TEST_CONTENTS.items():
        title = data["title"]
        content = data["content"]
        
        try:
            if REPORTLAB_AVAILABLE:
                filepath = create_pdf_with_reportlab(filename, title, content, output_path)
                print(f"✅ PDF erstellt: {filepath}")
            else:
                filepath = create_pdf_fallback(filename, title, content, output_path)
                print(f"✅ Text-Datei erstellt: {filepath}")
                
            created_files.append(filepath)
            
        except Exception as e:
            print(f"❌ Fehler bei {filename}: {e}")
    
    print(f"\n✅ {len(created_files)} Test-Dateien erstellt in {output_path}")
    
    # Test-Kommandos anzeigen
    print("\n📋 Test-Kommandos:")
    for filepath in created_files:
        print(f"   python cli.py pdf {filepath}")
        print(f"   python cli.py classify {filepath}")
    
    return created_files

def demo_pdf_generation():
    """Demo-Funktion für PDF-Generierung"""
    print("🎯 Demo: Test-PDF-Generierung")
    
    if REPORTLAB_AVAILABLE:
        print("✅ reportlab verfügbar - PDFs werden erstellt")
    else:
        print("⚠️  reportlab nicht verfügbar - Text-Fallback wird verwendet")
        print("   pip install reportlab für echte PDFs")
    
    files = generate_test_pdfs()
    
    if files:
        print(f"\n🎉 {len(files)} Test-Dateien erfolgreich erstellt!")
        print("   Jetzt kannst du die PDF-Demo ausführen:")
        print("   python cli.py demo")
    else:
        print("\n❌ Keine Dateien erstellt")

if __name__ == "__main__":
    demo_pdf_generation()
