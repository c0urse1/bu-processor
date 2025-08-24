#!/usr/bin/env python3
"""
Test für robuste Ingestion mit Klassifikations-Fehler-Simulation
=============================================================

Testet:
- Fall A: Normale Klassifikation → Job DONE
- Fall B: Simulierter Klassifikations-Fehler → Job FAILED mit Retry
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

# Add the bu_processor package to path
sys.path.insert(0, str(Path(__file__).parent))

from bu_processor.ingest import get_job_manager, ClassificationError
from bu_processor.core.config import get_config

# Aktiviere DRY_RUN für Tests
os.environ["DRY_RUN_INGEST"] = "true"

class MockClassifier:
    """Mock Classifier für Tests"""
    
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.model_dir = "test_model"
    
    def classify_text(self, text):
        if self.should_fail:
            # Fall B: Simuliere Klassifikations-Fehler
            return {
                "error": "Simulated classification failure for testing",
                "predicted_label": None,
                "confidence": None
            }
        else:
            # Fall A: Normale Klassifikation
            return {
                "predicted_label": "Antrag",
                "predicted_category": "Bewerbungsunterlagen", 
                "confidence": 0.95,
                "all_scores": {
                    "Antrag": 0.95,
                    "Vertrag": 0.03,
                    "Sonstiges": 0.02
                }
            }
    
    def get_available_labels(self):
        return ["Antrag", "Vertrag", "Rechnung", "Sonstiges"]

def create_test_pdf():
    """Erstelle eine Test-PDF-Datei"""
    test_content = """
    BEWERBUNGSANTRAG
    
    Hiermit beantrage ich die Stelle als Softwareentwickler.
    
    Mit freundlichen Grüßen
    Max Mustermann
    """
    
    # Erstelle temporäre Datei
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        return f.name

async def test_case_a_success():
    """Fall A: Klassifikation erfolgreich → Job DONE"""
    print("\n" + "="*60)
    print("🟢 FALL A: Normale Klassifikation (sollte COMPLETED werden)")
    print("="*60)
    
    # Setup
    job_manager = get_job_manager()
    job_manager.classifier = MockClassifier(should_fail=False)
    
    # Test-Datei erstellen
    test_file = create_test_pdf()
    
    try:
        # Job erstellen und verarbeiten
        job = job_manager.create_job(test_file, "test_success.pdf")
        print(f"📋 Job erstellt: {job.job_id}")
        
        # Job verarbeiten
        result_job = await job_manager.process_job(job.job_id)
        
        # Ergebnis prüfen
        print(f"📊 Job Status: {result_job.status}")
        print(f"📊 Retry Count: {result_job.retry_count}")
        print(f"📊 Error Message: {result_job.error_message}")
        
        if result_job.status == "completed":
            print("✅ SUCCESS: Job wurde erfolgreich abgeschlossen")
            if result_job.result:
                metadata = result_job.result.get("metadata", {})
                print(f"📊 Predicted Label: {metadata.get('predicted_label')}")
                print(f"📊 Confidence: {metadata.get('predicted_confidence')}")
        else:
            print("❌ FAILURE: Job sollte COMPLETED sein")
            
    except Exception as e:
        print(f"❌ EXCEPTION in Fall A: {e}")
    finally:
        # Cleanup
        try:
            os.unlink(test_file)
        except:
            pass

async def test_case_b_failure():
    """Fall B: Klassifikations-Fehler → Job FAILED mit Retry"""
    print("\n" + "="*60)
    print("🔴 FALL B: Simulierter Klassifikations-Fehler (sollte FAILED werden)")
    print("="*60)
    
    # Setup
    job_manager = get_job_manager()
    job_manager.classifier = MockClassifier(should_fail=True)
    
    # Test-Datei erstellen
    test_file = create_test_pdf()
    
    try:
        # Job erstellen und verarbeiten
        job = job_manager.create_job(test_file, "test_failure.pdf")
        print(f"📋 Job erstellt: {job.job_id}")
        
        # Job verarbeiten
        result_job = await job_manager.process_job(job.job_id)
        
        # Ergebnis prüfen
        print(f"📊 Job Status: {result_job.status}")
        print(f"📊 Retry Count: {result_job.retry_count}")
        print(f"📊 Max Retries: {result_job.max_retries}")
        print(f"📊 Error Message: {result_job.error_message}")
        
        if result_job.status == "failed":
            print("✅ SUCCESS: Job wurde korrekt als FAILED markiert")
            if result_job.retry_count > 0:
                print(f"✅ SUCCESS: Retry-System wurde {result_job.retry_count} mal versucht")
            else:
                print("⚠️  WARNING: Keine Retry-Versuche registriert")
        else:
            print(f"❌ FAILURE: Job sollte FAILED sein, ist aber {result_job.status}")
            
    except Exception as e:
        print(f"❌ EXCEPTION in Fall B: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            os.unlink(test_file)
        except:
            pass

def simulate_pdf_extraction(file_path):
    """Simuliere PDF Text-Extraktion für Tests"""
    return "Test PDF content - Bewerbungsantrag für Softwareentwickler Position."

async def main():
    """Haupttest-Funktion"""
    print("🚀 INGESTION ROBUSTNESS TEST")
    print("🧪 Testing Classification Error Handling & Retry System")
    print(f"🔧 DRY_RUN Mode: {os.getenv('DRY_RUN_INGEST', 'false')}")
    
    # Mock PDF extraction für Tests
    import bu_processor.pipeline.classifier as classifier_module
    classifier_module.extract_text_from_pdf = simulate_pdf_extraction
    
    try:
        # Test Fall A: Erfolgreiche Klassifikation
        await test_case_a_success()
        
        # Test Fall B: Klassifikations-Fehler
        await test_case_b_failure()
        
        print("\n" + "="*60)
        print("🏁 TESTS ABGESCHLOSSEN")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚡ Test durch Benutzer abgebrochen")
    except Exception as e:
        print(f"\n💥 Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
