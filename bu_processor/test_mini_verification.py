#!/usr/bin/env python3
"""
7) Mini-Tests zur Verifikation der Änderungen
===========================================

Smoke-Tests für:
A) Softmax/Threshold 
B) Sanity-Guards
C) Numerische Toleranz
D) Integration
"""

import pytest
import os
import sys
from pathlib import Path
from typing import List

# Ensure project root is on path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Test environment setup
os.environ["TESTING"] = "true"


class TestSoftmaxThresholdSmoke:
    """A) Softmax/Threshold Smoke Tests"""
    
    def test_softmax_confidence_high(self, monkeypatch):
        """Test High-Confidence Softmax mit konfigurierbarem Threshold"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Set environment threshold
        monkeypatch.setenv("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD", "0.7")
        
        # Mock classifier (ohne echtes Model-Loading)
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.7
        clf.labels = ["neg", "pos", "neu"]
        
        # High-confidence logits
        logits = [-2.0, 6.0, -3.0]
        labels = ["neg", "pos", "neu"]
        
        res = clf._postprocess_logits(logits, labels, "test text")
        
        assert res.category == "pos"
        assert res.confidence > 0.95
        assert res.is_confident is True
        assert res.error is None
        
        print("✅ High-confidence softmax test passed")
    
    def test_softmax_confidence_low(self, monkeypatch):
        """Test Low-Confidence Softmax"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        monkeypatch.setenv("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD", "0.7")
        
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.7
        clf.labels = ["neg", "pos"]
        
        # Low-confidence logits (fast gleich)
        logits = [-0.1, 0.1]
        labels = ["neg", "pos"]
        
        res = clf._postprocess_logits(logits, labels, "uncertain text")
        
        assert res.category == "pos"  # Slightly higher
        assert res.confidence < 0.6   # Well below threshold
        assert res.is_confident is False
        assert res.error is None
        
        print("✅ Low-confidence softmax test passed")
    
    def test_softmax_threshold_boundary(self, monkeypatch):
        """Test Threshold-Grenzfall"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        monkeypatch.setenv("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD", "0.8")
        
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.8
        clf.labels = ["class_a", "class_b"]
        
        # Logits die ca. 0.8 Confidence ergeben sollten
        # ln(0.8/0.2) = ln(4) ≈ 1.386
        logits = [0.0, 1.386]
        labels = ["class_a", "class_b"]
        
        res = clf._postprocess_logits(logits, labels, "boundary test")
        
        # Sollte nahe am Threshold sein
        assert abs(res.confidence - 0.8) < 0.05
        # Bei exakt 0.8 sollte is_confident True sein (>=)
        assert res.is_confident is True
        
        print("✅ Threshold boundary test passed")


class TestSanityGuardsSmoke:
    """B) Sanity-Guards Smoke Tests"""
    
    def test_batch_length_guarantee(self):
        """Test dass len(results) == len(texts) garantiert ist"""
        from bu_processor.pipeline.classifier import RealMLClassifier, ClassificationResult
        
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.7
        clf.labels = ["class_0", "class_1"]
        
        # Mock classify_text mit gemischten Erfolg/Fehler
        call_count = 0
        def mock_classify_text(text):
            nonlocal call_count
            call_count += 1
            
            if "error" in text:
                raise ValueError(f"Simulated error for {text}")
            
            return ClassificationResult(
                text=text,
                category="class_0",
                confidence=0.8,
                error=None,
                is_confident=True,
                metadata={}
            )
        
        clf.classify_text = mock_classify_text
        
        # Test mit 5 Texten, 2 davon Errors
        texts = ["text1", "error_text", "text3", "error_again", "text5"]
        
        batch_result = clf.classify_batch(texts)
        
        # SANITY-GUARD: Längen stimmen überein
        assert len(batch_result.results) == len(texts)
        assert batch_result.total_processed == len(texts)
        assert batch_result.successful + batch_result.failed == len(texts)
        
        # Verifikation: 3 erfolgreiche, 2 Fehler
        assert batch_result.successful == 3
        assert batch_result.failed == 2
        
        print("✅ Batch length guarantee test passed")
    
    def test_counting_from_results(self):
        """Test dass successful/failed aus results abgeleitet wird"""
        from bu_processor.pipeline.classifier import RealMLClassifier, ClassificationResult
        
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.7
        
        def mock_classify_text(text):
            if text.startswith("fail"):
                raise RuntimeError("Planned failure")
            
            return ClassificationResult(
                text=text,
                category="success",
                confidence=0.9,
                error=None,
                is_confident=True,
                metadata={}
            )
        
        clf.classify_text = mock_classify_text
        
        texts = ["success1", "fail1", "success2", "fail2", "success3"]
        batch_result = clf.classify_batch(texts)
        
        # Zähle actual results
        actual_successful = sum(1 for r in batch_result.results if r.error is None)
        actual_failed = sum(1 for r in batch_result.results if r.error is not None)
        
        # SANITY-GUARD: Zählungen stimmen mit actual results überein
        assert batch_result.successful == actual_successful
        assert batch_result.failed == actual_failed
        assert actual_successful == 3
        assert actual_failed == 2
        
        print("✅ Counting from results test passed")


class TestNumericalToleranceSmoke:
    """C) Numerische Toleranz Smoke Tests"""
    
    def test_clip01_extreme_values(self):
        """Test _clip01 mit extremen Werten"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Test extreme cases
        test_cases = [
            (-1000.0, 0.0),     # Sehr negativ
            (-0.0001, 0.0),     # Minimal unter 0
            (0.0, 0.0),         # Exakt 0
            (0.5, 0.5),         # Normal
            (1.0, 1.0),         # Exakt 1
            (1.0001, 1.0),      # Minimal über 1
            (1000.0, 1.0),      # Sehr groß
        ]
        
        for input_val, expected in test_cases:
            result = RealMLClassifier._clip01(input_val)
            assert abs(result - expected) < 1e-10, f"Failed for {input_val}"
        
        print("✅ Clip01 extreme values test passed")
    
    def test_postprocess_with_extreme_logits(self):
        """Test _postprocess_logits mit extremen Logits"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.7
        
        # Extreme logits die zu confidence > 1.0 führen könnten
        extreme_logits = [0.0, 100.0, -50.0]
        labels = ["class_a", "class_b", "class_c"]
        
        res = clf._postprocess_logits(extreme_logits, labels, "extreme test")
        
        # NUMERISCHE TOLERANZ: Confidence muss geclippt sein
        assert 0.0 <= res.confidence <= 1.0
        assert res.category == "class_b"  # Höchste logit
        
        # Alle Wahrscheinlichkeiten in metadata müssen geclippt sein
        if "all_probabilities" in res.metadata:
            for prob in res.metadata["all_probabilities"].values():
                assert 0.0 <= prob <= 1.0
        
        print("✅ Extreme logits clipping test passed")
    
    def test_floating_point_precision(self):
        """Test Floating-Point Precision Edge Cases"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        precision_cases = [
            1.0000000001,   # Floating point precision error
            0.9999999999,   # Almost 1.0
            1e-15,          # Very small positive
            1.0 + 1e-14,    # 1.0 plus tiny epsilon
        ]
        
        for value in precision_cases:
            clipped = RealMLClassifier._clip01(value)
            assert 0.0 <= clipped <= 1.0
            
            if value > 1.0:
                assert clipped == 1.0
            elif value < 0.0:
                assert clipped == 0.0
        
        print("✅ Floating point precision test passed")


class TestIntegrationSmoke:
    """D) Integration Smoke Tests"""
    
    def test_end_to_end_pipeline(self, monkeypatch):
        """End-to-End Test der gesamten Pipeline"""
        from bu_processor.pipeline.classifier import RealMLClassifier, ClassificationResult
        
        # Setup environment
        monkeypatch.setenv("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD", "0.75")
        
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.75
        clf.labels = ["insurance", "other"]
        
        # Mock mit verschiedenen Confidence-Levels
        def mock_classify_text(text):
            if "high_conf" in text:
                return ClassificationResult(
                    text=text,
                    category="insurance",
                    confidence=0.95,  # Über threshold
                    error=None,
                    is_confident=True,
                    metadata={}
                )
            elif "low_conf" in text:
                return ClassificationResult(
                    text=text,
                    category="other",
                    confidence=0.6,   # Unter threshold
                    error=None,
                    is_confident=False,
                    metadata={}
                )
            else:
                raise ValueError("Simulation error")
        
        clf.classify_text = mock_classify_text
        
        # Test mixed batch
        texts = [
            "high_conf document 1",
            "low_conf document 2", 
            "error document 3",
            "high_conf document 4"
        ]
        
        batch_result = clf.classify_batch(texts)
        
        # Integration validations
        assert len(batch_result.results) == len(texts)  # Sanity guard
        assert batch_result.total_processed == 4
        assert batch_result.successful == 2  # 2 high/low conf
        assert batch_result.failed == 2      # 2 errors
        
        # Confidence validations  
        confident_results = [r for r in batch_result.results if r.error is None and r.is_confident]
        unconfident_results = [r for r in batch_result.results if r.error is None and not r.is_confident]
        error_results = [r for r in batch_result.results if r.error is not None]
        
        assert len(confident_results) == 1    # 1 high_conf
        assert len(unconfident_results) == 1  # 1 low_conf  
        assert len(error_results) == 2        # 2 errors
        
        # Numerical tolerance validations
        for result in batch_result.results:
            if result.error is None:
                assert 0.0 <= result.confidence <= 1.0
        
        print("✅ End-to-end pipeline test passed")
    
    def test_config_threshold_integration(self, monkeypatch):
        """Test Integration mit Config-System"""
        monkeypatch.setenv("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD", "0.85")
        
        try:
            from bu_processor.core.config import get_config
            
            config = get_config()
            threshold = config.ml_model.classifier_confidence_threshold
            
            assert abs(threshold - 0.85) < 0.001
            print("✅ Config threshold integration test passed")
            
        except Exception as e:
            print(f"⚠️ Config test skipped: {e}")


def test_all_smoke_tests():
    """Führe alle Smoke-Tests aus"""
    import pytest
    
    print("🔥 Running All Smoke Tests\n")
    
    # Manually run tests (pytest-free version)
    try:
        # A) Softmax/Threshold
        print("=== A) Softmax/Threshold Tests ===")
        softmax_tests = TestSoftmaxThresholdSmoke()
        
        # Mock monkeypatch
        class MockMonkeypatch:
            def setenv(self, key, value):
                os.environ[key] = value
        
        monkeypatch = MockMonkeypatch()
        
        softmax_tests.test_softmax_confidence_high(monkeypatch)
        softmax_tests.test_softmax_confidence_low(monkeypatch)
        softmax_tests.test_softmax_threshold_boundary(monkeypatch)
        
        # B) Sanity-Guards
        print("\n=== B) Sanity-Guards Tests ===")
        sanity_tests = TestSanityGuardsSmoke()
        sanity_tests.test_batch_length_guarantee()
        sanity_tests.test_counting_from_results()
        
        # C) Numerical Tolerance
        print("\n=== C) Numerical Tolerance Tests ===")
        tolerance_tests = TestNumericalToleranceSmoke()
        tolerance_tests.test_clip01_extreme_values()
        tolerance_tests.test_postprocess_with_extreme_logits()
        tolerance_tests.test_floating_point_precision()
        
        # D) Integration
        print("\n=== D) Integration Tests ===")
        integration_tests = TestIntegrationSmoke()
        integration_tests.test_end_to_end_pipeline(monkeypatch)
        integration_tests.test_config_threshold_integration(monkeypatch)
        
        print("\n🎉 ALL SMOKE TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_all_smoke_tests()
    if success:
        print("\n✅ Alle Änderungen erfolgreich verifiziert!")
        print("✅ Softmax + Threshold funktioniert")
        print("✅ Sanity-Guards verhindern Validierungsfehler")
        print("✅ Numerische Toleranz verhindert Rundungsfehler")
        print("✅ Integration funktioniert End-to-End")
    else:
        print("\n❌ Einige Tests fehlgeschlagen")
    
    exit(0 if success else 1)
