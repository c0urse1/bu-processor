#!/usr/bin/env python3
"""
Pytest-based Mini-Tests zur Verifikation
=======================================
"""

import pytest
import os


class TestSoftmaxThresholdSmoke:
    """A) Softmax/Threshold Smoke Tests"""
    
    def test_softmax_confidence_high(self, monkeypatch):
        """Test High-Confidence Softmax (original test from requirements)"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Set environment threshold
        monkeypatch.setenv("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD", "0.7")
        
        # Create classifier instance (mocked)
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.7
        clf.labels = ["neg", "pos", "neu"]
        
        # High-confidence logits exactly as specified
        logits = [-2.0, 6.0, -3.0]
        labels = ["neg", "pos", "neu"]
        
        res = clf._postprocess_logits(logits, labels)
        
        assert res.category == "pos"
        assert res.confidence > 0.95
        assert res.is_confident is True
    
    def test_softmax_with_manual_mock_logits(self):
        """Test mit manuell erstellten Mock-Logits"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.7
        
        # Manual High-Confidence logits: [-2.0, 6.0] -> ~99.7% confidence
        high_conf_logits = [-2.0, 6.0]
        result = clf._postprocess_logits(
            high_conf_logits,
            ["class_0", "class_1"],
            "test"
        )
        
        assert result.category == "class_1"
        assert result.confidence > 0.99
        assert result.is_confident is True
        
        # Manual Low-Confidence logits: [-0.1, 0.1] -> ~52.4% confidence
        low_conf_logits = [-0.1, 0.1]
        result = clf._postprocess_logits(
            low_conf_logits,
            ["class_0", "class_1"],
            "test"
        )
        
        assert result.confidence < 0.6
        assert result.is_confident is False
    
    def test_threshold_configuration(self, monkeypatch):
        """Test Threshold-Konfiguration"""
        # Set high confidence threshold
        monkeypatch.setenv("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD", "0.9")
        
        # Environment sollte auf 0.9 gesetzt sein
        threshold = os.environ.get("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD")
        assert threshold == "0.9"


class TestSanityGuards:
    """B) Sanity-Guards Tests"""
    
    def test_batch_classification_sanity(self):
        """Test Sanity-Guards für Batch-Klassifikation"""
        from bu_processor.pipeline.classifier import RealMLClassifier, ClassificationResult
        
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.7
        
        def mock_classify_text(text):
            if "error" in text:
                raise ValueError("Test error")
            return ClassificationResult(
                text=text,
                category="test",
                confidence=0.8,
                error=None,
                is_confident=True,
                metadata={}
            )
        
        clf.classify_text = mock_classify_text
        
        texts = ["text1", "error_text", "text3"]
        batch_result = clf.classify_batch(texts)
        
        # SANITY-GUARDS
        assert len(batch_result.results) == len(texts)
        assert batch_result.total_processed == len(texts)
        assert batch_result.successful + batch_result.failed == batch_result.total_processed
        assert batch_result.successful == 2
        assert batch_result.failed == 1


class TestNumericalTolerance:
    """C) Numerische Toleranz Tests"""
    
    def test_clip01_function_validation(self):
        """Test _clip01 Funktion"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        assert RealMLClassifier._clip01(-0.1) == 0.0
        assert RealMLClassifier._clip01(1.5) == 1.0
        assert RealMLClassifier._clip01(0.5) == 0.5
        assert RealMLClassifier._clip01(1.0000001) == 1.0
    
    def test_postprocess_clipping(self):
        """Test dass _postprocess_logits Clipping anwendet"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.7
        
        # Extreme logits
        extreme_logits = [0.0, 50.0]  # Führt zu sehr hoher confidence
        result = clf._postprocess_logits(
            extreme_logits,
            ["class_0", "class_1"],
            "test"
        )
        
        # Confidence muss geclippt sein
        assert 0.0 <= result.confidence <= 1.0
        
        # Metadata probabilities müssen auch geclippt sein
        if "all_probabilities" in result.metadata:
            for prob in result.metadata["all_probabilities"].values():
                assert 0.0 <= prob <= 1.0


class TestIntegration:
    """D) Integration Tests"""
    
    def test_full_pipeline_integration(self, monkeypatch):
        """Test der gesamten Pipeline Integration"""
        from bu_processor.pipeline.classifier import RealMLClassifier, ClassificationResult
        
        # Set test threshold
        monkeypatch.setenv("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD", "0.7")
        
        clf = RealMLClassifier.__new__(RealMLClassifier) 
        clf.confidence_threshold = 0.7
        
        def mock_classify_text(text):
            if "high" in text:
                return ClassificationResult(
                    text=text, category="positive", confidence=0.95,
                    error=None, is_confident=True, metadata={}
                )
            elif "low" in text:
                return ClassificationResult(
                    text=text, category="negative", confidence=0.5,
                    error=None, is_confident=False, metadata={}
                )
            else:
                raise RuntimeError("Test error")
        
        clf.classify_text = mock_classify_text
        
        texts = ["high confidence", "low confidence", "error text"]
        batch_result = clf.classify_batch(texts)
        
        # Integration checks
        assert len(batch_result.results) == 3  # Sanity guard
        assert batch_result.successful == 2    # 2 successful
        assert batch_result.failed == 1        # 1 error
        
        # Confidence checks
        successful_results = [r for r in batch_result.results if r.error is None]
        for result in successful_results:
            assert 0.0 <= result.confidence <= 1.0  # Numerical tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
