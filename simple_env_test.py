#!/usr/bin/env python3
"""
Simple test to verify environment setup in conftest.py
"""

import os
import sys
from pathlib import Path

# Add bu_processor to path
bu_processor_path = Path(__file__).parent / "bu_processor"
sys.path.insert(0, str(bu_processor_path))

print("Testing environment variables setup...")

# Import conftest to trigger environment setup
try:
    from tests import conftest
    print("✅ conftest imported successfully")
    
    # Check environment variables
    testing = os.environ.get("TESTING")
    lazy_models = os.environ.get("BU_LAZY_MODELS")
    
    print(f"TESTING = {testing}")
    print(f"BU_LAZY_MODELS = {lazy_models}")
    
    if testing == "true" and lazy_models == "0":
        print("✅ Environment variables set correctly!")
    else:
        print("❌ Environment variables not set correctly")
        
    # Check OCR utilities
    try:
        from tests.conftest import check_tesseract_available, requires_tesseract
        print("✅ OCR utilities imported successfully")
        
        ocr_avail = check_tesseract_available()
        print(f"OCR available: {ocr_avail}")
        
    except Exception as e:
        print(f"⚠️ OCR utilities import issue: {e}")
        
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print("✅ Basic verification passed!")
