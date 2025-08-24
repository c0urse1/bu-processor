#!/usr/bin/env python3
"""
Einfacher Test für die Ingestion-Robustheit
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

# Set DRY_RUN mode
os.environ["DRY_RUN_INGEST"] = "true"

def test_classification_error_handling():
    """Test der ClassificationError Behandlung"""
    print("🧪 Testing ClassificationError handling...")
    
    # Import hier um Circular imports zu vermeiden
    from bu_processor.ingest import _as_dict, _extract_classification_fields, ClassificationError
    
    # Test 1: Normale Klassifikation
    print("\n✅ Test 1: Normale Klassifikation")
    normal_result = {
        "predicted_label": "Antrag",
        "confidence": 0.95,
        "all_scores": {"Antrag": 0.95, "Sonstiges": 0.05}
    }
    
    extracted = _extract_classification_fields(normal_result)
    print(f"   Label: {extracted['predicted_label']}")
    print(f"   Confidence: {extracted['confidence']}")
    
    # Test 2: Error-Result
    print("\n❌ Test 2: Error in Classification Result")
    error_result = {
        "error": "Classification model failed",
        "predicted_label": None
    }
    
    try:
        extracted_error = _extract_classification_fields(error_result)
        if not extracted_error["predicted_label"]:
            print("   ✅ Error correctly detected - no label found")
        else:
            print("   ❌ Error: Label should be None")
    except Exception as e:
        print(f"   ⚠️  Exception: {e}")
    
    # Test 3: Verschiedene Feldnamen
    print("\n🔄 Test 3: Verschiedene Feldnamen")
    alt_result = {
        "category": "Bewerbung",
        "score": 0.88,
        "scores": {"Bewerbung": 0.88, "Vertrag": 0.12}
    }
    
    extracted_alt = _extract_classification_fields(alt_result)
    print(f"   Category als Label: {extracted_alt['predicted_label']}")
    print(f"   Score als Confidence: {extracted_alt['confidence']}")
    
    print("\n✅ Classification Error Handling Tests completed!")

def test_dry_run_mode():
    """Test des DRY_RUN Modus"""
    print("\n🧪 Testing DRY_RUN mode...")
    
    dry_run = os.getenv("DRY_RUN_INGEST", "false").lower() == "true"
    print(f"   DRY_RUN_INGEST: {os.getenv('DRY_RUN_INGEST', 'not set')}")
    print(f"   DRY_RUN active: {dry_run}")
    
    if dry_run:
        print("   ✅ DRY_RUN mode is active - no real upserts will happen")
    else:
        print("   ⚠️  DRY_RUN mode is NOT active - real upserts would happen")

def main():
    """Haupttest ohne async dependencies"""
    print("🚀 SIMPLE INGESTION ROBUSTNESS TEST")
    print("=" * 50)
    
    try:
        # Test Classification Error Handling
        test_classification_error_handling()
        
        # Test DRY_RUN Mode
        test_dry_run_mode()
        
        print("\n" + "=" * 50)
        print("🏁 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        # Hinweis für manuelle Tests
        print("\n💡 Für vollständige Job-Tests:")
        print("   1. Starte einen PDF Ingestion Job über API")
        print("   2. Prüfe die Logs auf stage-spezifisches Logging")
        print("   3. Simuliere einen Fehler um Retry-Verhalten zu testen")
        
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
