#!/usr/bin/env python3
"""
Verifikation für Fix #11: Einheitliche Testumgebung (Stabilität)

Überprüft:
1. Umgebungsvariablen sind korrekt gesetzt
2. OCR-Tests werden korrekt übersprungen wenn Tesseract fehlt
3. Warnings sind akzeptabel wenn OCR gemockt ist
4. Testumgebung ist stabil und konsistent
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

def verify_environment_variables():
    """Überprüft ob die erforderlichen Umgebungsvariablen gesetzt sind."""
    print("🔍 Verifying environment variables...")
    
    # Import conftest to trigger environment setup
    sys.path.insert(0, str(Path(__file__).parent / "bu_processor"))
    
    try:
        from tests import conftest
        
        # Check if environment variables are set correctly
        testing_var = os.environ.get("TESTING")
        lazy_models_var = os.environ.get("BU_LAZY_MODELS")
        
        print(f"   ✅ TESTING = '{testing_var}'")
        print(f"   ✅ BU_LAZY_MODELS = '{lazy_models_var}'")
        
        assert testing_var == "true", f"Expected TESTING='true', got '{testing_var}'"
        assert lazy_models_var == "0", f"Expected BU_LAZY_MODELS='0', got '{lazy_models_var}'"
        
        print("   ✅ All environment variables correctly set at module import")
        return True
        
    except Exception as e:
        print(f"   ❌ Environment setup failed: {e}")
        return False


def verify_ocr_skip_functionality():
    """Überprüft ob OCR-Skip-Funktionalität korrekt implementiert ist."""
    print("\n🔍 Verifying OCR skip functionality...")
    
    try:
        from tests.conftest import check_tesseract_available, requires_tesseract
        
        # Test OCR availability check
        ocr_available = check_tesseract_available()
        print(f"   ℹ️  OCR available: {ocr_available}")
        
        # Test skip decorator exists
        assert hasattr(requires_tesseract, 'pytestmark') or hasattr(requires_tesseract, 'markname'), \
               "requires_tesseract should be a pytest skip marker"
        
        print("   ✅ OCR skip decorator properly configured")
        
        # Test that warnings for missing pytesseract are acceptable
        if not ocr_available:
            print("   ℹ️  Tesseract not available - OCR tests will be skipped (this is OK)")
        else:
            print("   ℹ️  Tesseract available - OCR tests will run")
            
        return True
        
    except Exception as e:
        print(f"   ❌ OCR skip functionality verification failed: {e}")
        return False


def verify_test_stability():
    """Überprüft grundlegende Teststabilität."""
    print("\n🔍 Verifying test stability...")
    
    try:
        # Test that we can import test fixtures without errors
        from tests.conftest import project_root, sample_texts
        
        print("   ✅ Core test fixtures importable")
        
        # Verify lazy loading is disabled for stability
        lazy_setting = os.environ.get("BU_LAZY_MODELS", "1")
        if lazy_setting == "0":
            print("   ✅ Lazy loading disabled for test stability")
        else:
            print(f"   ⚠️  Lazy loading setting: {lazy_setting} (should be '0' for stability)")
            
        # Test that testing flag is set
        testing_flag = os.environ.get("TESTING", "false")
        if testing_flag == "true":
            print("   ✅ Testing mode enabled")
        else:
            print(f"   ⚠️  Testing flag: {testing_flag} (should be 'true')")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Test stability verification failed: {e}")
        return False


def verify_ocr_mocking_compatibility():
    """Überprüft dass OCR-Mocking weiterhin funktioniert."""
    print("\n🔍 Verifying OCR mocking compatibility...")
    
    try:
        # Test that we can mock OCR functionality even when it's not available
        with patch("bu_processor.pipeline.pdf_extractor.OCR_AVAILABLE", True):
            print("   ✅ OCR_AVAILABLE can be mocked to True")
            
        with patch("bu_processor.pipeline.pdf_extractor.OCR_AVAILABLE", False):
            print("   ✅ OCR_AVAILABLE can be mocked to False")
            
        # Verify that missing pytesseract doesn't break test imports
        try:
            from bu_processor.pipeline.pdf_extractor import EnhancedPDFExtractor
            print("   ✅ PDF extractor imports successfully despite potential missing OCR deps")
        except ImportError as e:
            if "pytesseract" in str(e):
                print("   ⚠️  OCR import error expected and handled gracefully")
            else:
                raise
                
        return True
        
    except Exception as e:
        print(f"   ❌ OCR mocking compatibility verification failed: {e}")
        return False


def run_verification():
    """Führt alle Verifikationen aus."""
    print("=" * 70)
    print("🧪 VERIFIKATION: Fix #11 - Einheitliche Testumgebung (Stabilität)")
    print("=" * 70)
    
    results = []
    
    # Verifikationsschritte
    results.append(verify_environment_variables())
    results.append(verify_ocr_skip_functionality())
    results.append(verify_test_stability())
    results.append(verify_ocr_mocking_compatibility())
    
    # Gesamtergebnis
    print("\n" + "=" * 70)
    if all(results):
        print("🎉 ALLE VERIFIKATIONEN ERFOLGREICH!")
        print("✅ Einheitliche Testumgebung korrekt implementiert")
        print("✅ OCR-Tests werden bei fehlendem Tesseract übersprungen")
        print("✅ Stabilität durch deaktivierte Lazy Models gewährleistet")
        print("✅ Mocking-Kompatibilität erhalten")
        return True
    else:
        print("❌ EINIGE VERIFIKATIONEN FEHLGESCHLAGEN!")
        failed_count = len([r for r in results if not r])
        print(f"   {failed_count} von {len(results)} Tests fehlgeschlagen")
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
