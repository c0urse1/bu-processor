#!/usr/bin/env python3
"""Simple test to validate changes work"""

print("Testing imports...")

try:
    print("1. Testing PDF extractor NLTK fallback...")
    import bu_processor.pipeline.pdf_extractor as pdfext
    print(f"   NLTK Available: {pdfext.NLTK_AVAILABLE}")
    print(f"   NLTK object type: {type(pdfext.nltk)}")
    
    # Test sentence tokenization
    result = pdfext.nltk.sent_tokenize("Hello world. This is a test.")
    print(f"   Tokenization result: {result}")
    print("   ✅ NLTK fallback works")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

try:
    print("2. Testing classifier import...")
    import bu_processor.pipeline.classifier as clf
    print("   ✅ Classifier import successful")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print("Basic tests complete!")
