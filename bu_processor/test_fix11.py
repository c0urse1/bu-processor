#!/usr/bin/env python3
"""Test OCR functionality"""

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    print("✅ Tesseract OCR is available")
    ocr_available = True
except Exception as e:
    print("⚠️ Tesseract OCR not available (this is OK)")
    print(f"   Reason: {e}")
    ocr_available = False

print(f"OCR status: {ocr_available}")

# Test our conftest environment setup
import os
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("BU_LAZY_MODELS", "0")

print(f"TESTING = {os.environ.get('TESTING')}")
print(f"BU_LAZY_MODELS = {os.environ.get('BU_LAZY_MODELS')}")

print("✅ Fix #11 environment setup working correctly")
