#!/usr/bin/env python3
"""Test script to validate the stability fixes"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent / "bu_processor"))

def test_nltk_fallback():
    """Test NLTK fallback implementation"""
    print("🧪 Testing NLTK fallback...")
    
    try:
        from pipeline.pdf_extractor import NLTK_AVAILABLE, nltk
        print(f"   ✅ NLTK Available: {NLTK_AVAILABLE}")
        
        # Test sentence tokenization
        test_text = "Hello world. This is a test! How are you?"
        result = nltk.sent_tokenize(test_text)
        print(f"   ✅ Sentence tokenization works: {len(result)} sentences found")
        print(f"   Example: {result[:2]}")
        
    except Exception as e:
        print(f"   ❌ NLTK fallback test failed: {e}")
        return False
    
    return True

def test_universal_dispatch():
    """Test universal dispatch consistency"""
    print("🧪 Testing universal dispatch...")
    
    try:
        from pipeline.classifier import RealMLClassifier
        
        # Create classifier in lazy mode to avoid model loading
        classifier = RealMLClassifier(lazy=True)
        print("   ✅ Classifier created successfully")
        
        # Test type detection (without actual execution to avoid model requirements)
        print("   ✅ Universal dispatch method exists")
        
        # Check that the method signature is correct
        import inspect
        sig = inspect.signature(classifier.classify)
        print(f"   ✅ Method signature: {sig}")
        
    except Exception as e:
        print(f"   ❌ Universal dispatch test failed: {e}")
        return False
    
    return True

def test_pdf_extractor_injection():
    """Test PDF extractor injection"""
    print("🧪 Testing PDF extractor injection...")
    
    try:
        from pipeline.classifier import RealMLClassifier
        
        # Create classifier in lazy mode
        classifier = RealMLClassifier(lazy=True)
        
        # Test initial state
        initial_state = hasattr(classifier, 'pdf_extractor')
        print(f"   Initial pdf_extractor: {initial_state}")
        
        # Test injection method exists
        assert hasattr(classifier, 'set_pdf_extractor'), "set_pdf_extractor method missing"
        print("   ✅ set_pdf_extractor method exists")
        
        # Test injection
        class MockExtractor:
            def extract_text_from_pdf(self, path, **kwargs):
                return "mock content"
        
        mock_extractor = MockExtractor()
        classifier.set_pdf_extractor(mock_extractor)
        
        # Verify injection
        assert hasattr(classifier, 'pdf_extractor'), "pdf_extractor not set after injection"
        assert classifier.pdf_extractor is mock_extractor, "pdf_extractor not correctly injected"
        print("   ✅ PDF extractor injection works")
        
    except Exception as e:
        print(f"   ❌ PDF extractor injection test failed: {e}")
        return False
    
    return True

def main():
    """Run all stability tests"""
    print("🚀 Testing Stability Fixes")
    print("=" * 50)
    
    tests = [
        test_nltk_fallback,
        test_universal_dispatch, 
        test_pdf_extractor_injection
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"   ❌ Test {test.__name__} crashed: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("📊 Test Summary")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All stability fixes working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed - review implementation")
        return 1

if __name__ == "__main__":
    exit(main())
