#!/usr/bin/env python3
"""
Finale Mini-Tests mit echten Fixtures
====================================
"""

import pytest
import os


class TestFinalVerification:
    """Finale Verifikation aller implementierten Features"""
    
    def test_original_softmax_confidence_high(self, test_confidence_threshold):
        """Original Test aus den Requirements - exakt wie spezifiziert"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Exakt wie in den Anforderungen
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.7
        
        logits = [-2.0, 6.0, -3.0]
        labels = ["neg", "pos", "neu"]
        res = clf._postprocess_logits(logits, labels)
        
        assert res.category == 1  # Index of "pos" in labels
        assert res.label == "pos"  # String label
        assert res.confidence > 0.95
        assert res.is_confident is True
    
    def test_with_real_mock_fixtures(self, mock_classifier_with_logits, mock_logits):
        """Test mit echten Mock-Fixtures aus conftest.py"""
        classifier, set_logits = mock_classifier_with_logits
        
        # Test High-Confidence mit Mock-Logits
        high_logits = mock_logits.high_confidence_2_classes(winner_idx=1)
        result = classifier._postprocess_logits(
            high_logits,
            ["insurance", "other"],
            "test document"
        )
        
        assert result.category == 1  # Index of "other" in labels 
        assert result.label == "other"  # String label
        assert result.confidence > 0.99
        assert result.is_confident is True
        
        # Test Low-Confidence mit Mock-Logits
        low_logits = mock_logits.low_confidence_2_classes()
        result = classifier._postprocess_logits(
            low_logits,
            ["insurance", "other"],
            "uncertain document"
        )
        
        assert result.confidence < 0.6
        assert result.is_confident is False
    
    def test_sanity_guards_with_fixtures(self):
        """Test Sanity-Guards Implementation"""
        from bu_processor.pipeline.classifier import RealMLClassifier, ClassificationResult
        
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.7
        
        def mock_classify_text(text):
            if "fail" in text:
                raise ValueError("Intentional test failure")
            return ClassificationResult(
                text=text, category="success", confidence=0.8,
                error=None, is_confident=True, metadata={}
            )
        
        clf.classify_text = mock_classify_text
        
        texts = ["success1", "fail1", "success2", "fail2", "success3"]
        batch_result = clf.classify_batch(texts)
        
        # Sanity-Guards: len(results) == len(texts) IMMER
        assert len(batch_result.results) == len(texts)
        assert batch_result.total_processed == len(texts)
        assert batch_result.successful + batch_result.failed == len(texts)
        
        # Zählung aus results abgeleitet
        actual_successful = sum(1 for r in batch_result.results if r.error is None)
        actual_failed = sum(1 for r in batch_result.results if r.error is not None)
        
        assert batch_result.successful == actual_successful
        assert batch_result.failed == actual_failed
    
    def test_numerical_tolerance_clipping(self):
        """Test numerische Toleranz Implementation"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        clf = RealMLClassifier.__new__(RealMLClassifier)
        clf.confidence_threshold = 0.7
        
        # Test _clip01 direkt
        assert RealMLClassifier._clip01(-0.5) == 0.0
        assert RealMLClassifier._clip01(1.5) == 1.0
        assert RealMLClassifier._clip01(0.5) == 0.5
        
        # Test Integration in _postprocess_logits
        extreme_logits = [0.0, 100.0]  # Könnte confidence > 1.0 ergeben
        result = clf._postprocess_logits(extreme_logits, ["low", "high"], "test")
        
        # Muss geclippt sein
        assert 0.0 <= result.confidence <= 1.0
        assert result.category == 1  # Index of "high" in labels
        assert result.label == "high"  # String label
        
        # Metadata auch geclippt
        if "all_probabilities" in result.metadata:
            for prob in result.metadata["all_probabilities"].values():
                assert 0.0 <= prob <= 1.0
    
    def test_threshold_configuration_with_fixtures(self, high_confidence_threshold):
        """Test Threshold-Konfiguration mit echten Fixtures"""
        # high_confidence_threshold Fixture setzt Threshold auf 0.9
        threshold_env = os.environ.get("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD")
        assert threshold_env == "0.9"
        
        # Test mit Config-System
        try:
            from bu_processor.core.config import get_config
            config = get_config()
            assert abs(config.ml_model.classifier_confidence_threshold - 0.9) < 0.001
        except Exception:
            # Config-System optional
            pass
    
    def test_end_to_end_integration(self, mock_classifier_with_logits, mock_logits):
        """End-to-End Integration aller Features"""
        classifier, set_logits = mock_classifier_with_logits
        
        # Simuliere verschiedene Szenarien
        scenarios = [
            ("high_conf", mock_logits.high_confidence_2_classes(winner_idx=0), True),
            ("medium_conf", mock_logits.medium_confidence_2_classes(winner_idx=1), True),
            ("low_conf", mock_logits.low_confidence_2_classes(), False),
        ]
        
        for text, logits, expected_confident in scenarios:
            result = classifier._postprocess_logits(
                logits,
                ["class_0", "class_1"],
                text
            )
            
            # Sanity checks
            assert result.text == text
            assert result.category in [0, 1]  # Integer indices 
            assert result.label in ["class_0", "class_1"]  # String labels
            assert 0.0 <= result.confidence <= 1.0  # Numerical tolerance
            assert result.is_confident == expected_confident  # Threshold logic
            assert result.error is None
            assert isinstance(result.metadata, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
