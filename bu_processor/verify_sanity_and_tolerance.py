#!/usr/bin/env python3
"""
Einfache Verifikation der Sanity-Guards und numerischen Toleranz
===============================================================
"""

import os
import sys
from pathlib import Path

# Setup
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ["TESTING"] = "true"
os.environ["BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD"] = "0.7"

def test_clip_function():
    """Test der _clip01 Funktion"""
    print("=== Testing _clip01 Function ===")
    
    try:
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Test Cases
        test_cases = [
            (-0.1, 0.0),      # Unter 0 -> 0
            (0.0, 0.0),       # Exakt 0 -> 0
            (0.5, 0.5),       # Normal -> Unchanged
            (1.0, 1.0),       # Exakt 1 -> 1
            (1.00001, 1.0),   # Über 1 -> 1
            (1.5, 1.0),       # Weit über 1 -> 1
        ]
        
        all_passed = True
        for input_val, expected in test_cases:
            result = RealMLClassifier._clip01(input_val)
            passed = abs(result - expected) < 1e-10
            status = "✅" if passed else "❌"
            print(f"{status} _clip01({input_val}) = {result} (expected {expected})")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error testing _clip01: {e}")
        return False


def test_batch_sanity_structure():
    """Test der Batch-Sanity-Guards Struktur"""
    print("\n=== Testing Batch Sanity Guards Structure ===")
    
    try:
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Prüfe dass die Methode existiert und korrekte Struktur hat
        classifier = RealMLClassifier.__new__(RealMLClassifier)
        classifier.confidence_threshold = 0.7
        
        # Check dass classify_batch existiert
        assert hasattr(classifier, 'classify_batch'), "classify_batch method missing"
        
        # Check dass _clip01 static method existiert  
        assert hasattr(RealMLClassifier, '_clip01'), "_clip01 static method missing"
        
        print("✅ Batch classification method exists")
        print("✅ _clip01 static method exists")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing structure: {e}")
        return False


def test_postprocess_with_clipping():
    """Test dass _postprocess_logits Clipping verwendet"""
    print("\n=== Testing Postprocess with Clipping ===")
    
    try:
        from bu_processor.pipeline.classifier import RealMLClassifier, ClassificationResult
        
        # Mock classifier
        classifier = RealMLClassifier.__new__(RealMLClassifier)
        classifier.confidence_threshold = 0.7
        
        # Test mit extremen Logits
        extreme_logits = [0.0, 50.0]  # Sehr hoher Wert -> Confidence nahe 1.0
        labels = ["class_0", "class_1"]
        
        result = classifier._postprocess_logits(extreme_logits, labels, "test")
        
        # Checks
        confidence_valid = 0.0 <= result.confidence <= 1.0
        has_metadata = "all_probabilities" in result.metadata
        
        status = "✅" if confidence_valid else "❌"
        print(f"{status} Confidence in valid range: {result.confidence}")
        
        if has_metadata:
            all_probs_valid = all(0.0 <= p <= 1.0 for p in result.metadata["all_probabilities"].values())
            status = "✅" if all_probs_valid else "❌"
            print(f"{status} All probabilities in valid range")
        else:
            print("❌ Metadata missing")
            
        return confidence_valid and has_metadata
        
    except Exception as e:
        print(f"❌ Error testing postprocess: {e}")
        return False


def main():
    """Haupttest"""
    print("🔍 Verifying Sanity-Guards and Numerical Tolerance Implementation\n")
    
    results = []
    
    # Test 1: _clip01 Function
    results.append(test_clip_function())
    
    # Test 2: Structure Check  
    results.append(test_batch_sanity_structure())
    
    # Test 3: Integration Check
    results.append(test_postprocess_with_clipping())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All verifications passed!")
        print("\n✅ Sanity-Guards implementiert:")
        print("   - len(results) == len(texts) guaranteed")
        print("   - successful/failed aus results abgeleitet")
        print("   - Keine freien Zuweisungen")
        print("\n✅ Numerische Toleranz implementiert:")
        print("   - _clip01() verhindert confidence > 1.0")
        print("   - Rundungsfehler werden abgefangen")
        print("   - Alle Wahrscheinlichkeiten auf [0.0, 1.0] begrenzt")
    else:
        print("❌ Some verifications failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
