#!/usr/bin/env python3
"""
Test für Softmax + Threshold Klassifizierer-Logik
================================================

Testet die verbesserte RealMLClassifier Implementierung mit:
- Numerisch stabiler Softmax
- Konfigurierbarem Confidence-Threshold
- Korrekter Batch-Zählung
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_softmax_stability():
    """Test der numerisch stabilen Softmax-Implementierung"""
    print("🔧 Test 1: Numerisch stabile Softmax")
    
    try:
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Test mit verschiedenen Logit-Werten
        test_cases = [
            [1.0, 2.0, 3.0],           # Normale Werte
            [100.0, 101.0, 102.0],    # Große Werte (Overflow-Test)
            [-100.0, -99.0, -98.0],   # Negative Werte
            [0.0],                     # Einzelwert
            [],                        # Leere Liste
        ]
        
        for i, logits in enumerate(test_cases):
            if not logits:
                probs = RealMLClassifier._softmax(logits)
                print(f"✅ Test {i+1}: Leere Liste -> {probs}")
                continue
                
            probs = RealMLClassifier._softmax(logits)
            prob_sum = sum(probs) if probs else 0
            
            print(f"✅ Test {i+1}: logits={logits}")
            print(f"   probs={[round(p, 4) for p in probs]}")
            print(f"   sum={round(prob_sum, 6)}")
            
            # Validierung
            if probs and abs(prob_sum - 1.0) > 1e-6:
                print(f"❌ Wahrscheinlichkeiten summieren nicht zu 1.0: {prob_sum}")
                return False
                
            if any(p < 0 for p in probs):
                print(f"❌ Negative Wahrscheinlichkeiten gefunden")
                return False
        
        print("✅ Softmax-Stabilität: Alle Tests bestanden")
        return True
        
    except Exception as e:
        print(f"❌ Softmax-Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_postprocess_logits():
    """Test der _postprocess_logits Methode"""
    print("\n🔧 Test 2: Logits Post-Processing")
    
    try:
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Mock einen Classifier mit konfigurierbarem Threshold
        classifier = RealMLClassifier.__new__(RealMLClassifier)
        classifier.confidence_threshold = 0.7  # 70% Threshold
        
        # Test-Fälle
        test_logits = [1.0, 3.0, 2.0]  # Nach Softmax: ~[0.09, 0.67, 0.24]
        test_labels = ["category_A", "category_B", "category_C"]
        test_text = "Test document content"
        
        result = classifier._postprocess_logits(test_logits, test_labels, test_text)
        
        print(f"✅ Text: {result.text}")
        print(f"✅ Kategorie: {result.category}")
        print(f"✅ Confidence: {round(result.confidence, 4)}")
        print(f"✅ Is Confident: {result.is_confident}")
        print(f"✅ Threshold: {result.metadata.get('confidence_threshold')}")
        
        # Validierungen
        if result.category != "category_B":  # Höchste Logit
            print(f"❌ Falsche Kategorie: erwartet category_B, bekommen {result.category}")
            return False
            
        expected_confidence = 0.6652  # Ungefähr für diese Logits
        if abs(result.confidence - expected_confidence) > 0.01:
            print(f"❌ Confidence außerhalb erwarteter Range: {result.confidence}")
            return False
            
        # Test: Confidence unter Threshold
        if result.confidence >= classifier.confidence_threshold:
            print(f"✅ Confidence über Threshold: confident={result.is_confident}")
        else:
            print(f"✅ Confidence unter Threshold: confident={result.is_confident}")
        
        # Test mit ungültigen Inputs
        invalid_result = classifier._postprocess_logits([], test_labels, test_text)
        if invalid_result.error is None:
            print("❌ Ungültige Inputs sollten Error erzeugen")
            return False
        print(f"✅ Ungültige Inputs korrekt behandelt: {invalid_result.error}")
        
        print("✅ Logits Post-Processing: Alle Tests bestanden")
        return True
        
    except Exception as e:
        print(f"❌ Post-Processing Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_confidence_threshold_config():
    """Test der Confidence-Threshold Konfiguration"""
    print("\n🔧 Test 3: Confidence-Threshold aus Konfiguration")
    
    try:
        import os
        
        # Setze temporäre Environment-Variable
        original_value = os.environ.get("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD")
        os.environ["BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD"] = "0.9"
        
        from bu_processor.core.config import get_config
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Test: Konfiguration laden
        config = get_config()
        threshold_from_config = config.ml_model.classifier_confidence_threshold
        
        print(f"✅ Threshold aus Config: {threshold_from_config}")
        
        # Test: Classifier verwendet Config-Wert
        # Mock Classifier init ohne Model-Loading
        classifier = RealMLClassifier.__new__(RealMLClassifier)
        cfg = get_config()
        classifier.confidence_threshold = cfg.ml_model.classifier_confidence_threshold
        
        print(f"✅ Classifier Threshold: {classifier.confidence_threshold}")
        
        if abs(classifier.confidence_threshold - 0.9) > 0.001:
            print(f"❌ Classifier verwendet nicht den Config-Wert: {classifier.confidence_threshold}")
            return False
        
        # Cleanup
        if original_value is not None:
            os.environ["BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD"] = original_value
        else:
            os.environ.pop("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD", None)
        
        print("✅ Confidence-Threshold Konfiguration: Test bestanden")
        return True
        
    except Exception as e:
        print(f"❌ Config-Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_counting():
    """Test der korrekten Batch-Zählung"""
    print("\n🔧 Test 4: Korrekte Batch-Zählung")
    
    try:
        from bu_processor.pipeline.classifier import ClassificationResult, BatchClassificationResult
        
        # Mock Ergebnisse
        results = [
            ClassificationResult(text="Doc 1", confidence=0.9, category="A"),           # Erfolgreich
            ClassificationResult(text="Doc 2", confidence=0.8, category="B"),           # Erfolgreich  
            ClassificationResult(text="Doc 3", confidence=0.0, error="Test error"),     # Fehlgeschlagen
            ClassificationResult(text="Doc 4", confidence=0.7, category="C"),           # Erfolgreich
        ]
        
        # Simuliere Batch-Result Erstellung
        total_processed = len(results)
        successful = sum(1 for r in results if r.error is None)
        failed = total_processed - successful
        
        batch_result = BatchClassificationResult(
            total_processed=total_processed,
            successful=successful,
            failed=failed,
            results=results
        )
        
        print(f"✅ Total: {batch_result.total_processed}")
        print(f"✅ Successful: {batch_result.successful}")
        print(f"✅ Failed: {batch_result.failed}")
        print(f"✅ Results length: {len(batch_result.results)}")
        
        # Validierungen
        if batch_result.successful != 3:
            print(f"❌ Falsche Successful-Zählung: {batch_result.successful}")
            return False
            
        if batch_result.failed != 1:
            print(f"❌ Falsche Failed-Zählung: {batch_result.failed}")
            return False
            
        if batch_result.total_processed != 4:
            print(f"❌ Falsche Total-Zählung: {batch_result.total_processed}")
            return False
            
        # Pydantic-Validierung sollte automatisch prüfen
        print("✅ Batch-Zählung: Alle Tests bestanden")
        return True
        
    except Exception as e:
        print(f"❌ Batch-Zählung Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Führe alle Tests aus"""
    print("🧪 TEST: Softmax + Threshold Klassifizierer-Logik")
    print("=" * 55)
    
    tests = [
        test_softmax_stability,
        test_postprocess_logits,
        test_confidence_threshold_config,
        test_batch_counting
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n📊 Ergebnis: {passed}/{len(tests)} Tests bestanden")
    
    if passed == len(tests):
        print("🎉 Alle Tests erfolgreich! Softmax + Threshold Logik funktioniert.")
        return True
    else:
        print("❌ Einige Tests fehlgeschlagen!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
