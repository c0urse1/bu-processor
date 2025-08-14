#!/usr/bin/env python3
"""
High-Confidence Tests mit Fixtures & Mocks
==========================================

Testet zuverlässige High-Confidence Klassifikationen mit:
- Konfigurierbaren Threshold-Fixtures
- Mock-Logits für garantierte Confidence-Werte
- Robuste Test-Umgebung
"""

import pytest
import math
from typing import List


class TestConfidenceThresholdFixtures:
    """Tests für Confidence-Threshold Fixtures"""
    
    def test_default_confidence_threshold(self):
        """Test dass default Threshold für Tests auf 0.7 gesetzt ist"""
        import os
        threshold = os.environ.get("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD")
        assert threshold == "0.7", f"Expected default test threshold 0.7, got {threshold}"
    
    def test_low_confidence_threshold_fixture(self, low_confidence_threshold):
        """Test der low_confidence_threshold Fixture"""
        import os
        threshold = os.environ.get("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD")
        assert threshold == "0.5", f"Expected low threshold 0.5, got {threshold}"
    
    def test_high_confidence_threshold_fixture(self, high_confidence_threshold):
        """Test der high_confidence_threshold Fixture"""
        import os
        threshold = os.environ.get("BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD")
        assert threshold == "0.9", f"Expected high threshold 0.9, got {threshold}"


class TestMockLogitsProvider:
    """Tests für MockLogitsProvider"""
    
    def test_mock_logits_fixture(self, mock_logits):
        """Test dass mock_logits Fixture verfügbar ist"""
        assert hasattr(mock_logits, 'high_confidence_2_classes')
        assert hasattr(mock_logits, 'high_confidence_3_classes')
        assert hasattr(mock_logits, 'medium_confidence_2_classes')
        assert hasattr(mock_logits, 'low_confidence_2_classes')
    
    def test_high_confidence_2_classes(self, mock_logits):
        """Test High-Confidence Logits für 2 Klassen"""
        # Test winner_idx = 1
        logits = mock_logits.high_confidence_2_classes(winner_idx=1)
        assert len(logits) == 2
        assert logits == [-2.0, 6.0]
        
        # Verifiziere Softmax-Ergebnis
        assert mock_logits.verify_softmax_confidence(logits, 0.997, tolerance=0.01)
        
        # Test winner_idx = 0
        logits = mock_logits.high_confidence_2_classes(winner_idx=0)
        assert logits == [6.0, -2.0]
        assert mock_logits.verify_softmax_confidence(logits, 0.997, tolerance=0.01)
    
    def test_high_confidence_3_classes(self, mock_logits):
        """Test High-Confidence Logits für 3 Klassen"""
        logits = mock_logits.high_confidence_3_classes(winner_idx=1)
        assert len(logits) == 3
        assert logits == [-2.0, 6.0, -2.0]
        
        # Verifiziere Softmax-Ergebnis
        assert mock_logits.verify_softmax_confidence(logits, 0.982, tolerance=0.01)
    
    def test_medium_confidence_2_classes(self, mock_logits):
        """Test Medium-Confidence Logits für 2 Klassen"""
        logits = mock_logits.medium_confidence_2_classes(winner_idx=1)
        assert len(logits) == 2
        assert logits == [0.0, 1.0]
        
        # Verifiziere Softmax-Ergebnis
        assert mock_logits.verify_softmax_confidence(logits, 0.731, tolerance=0.01)
    
    def test_low_confidence_2_classes(self, mock_logits):
        """Test Low-Confidence Logits für 2 Klassen"""
        logits = mock_logits.low_confidence_2_classes()
        assert len(logits) == 2
        assert logits == [-0.1, 0.1]
        
        # Verifiziere Softmax-Ergebnis
        assert mock_logits.verify_softmax_confidence(logits, 0.524, tolerance=0.01)
    
    def test_softmax_verification_utility(self, mock_logits):
        """Test der Softmax-Verifikations-Utility"""
        # Test mit bekannten Werten
        logits = [0.0, 1.0]  # Sollte ~0.731 ergeben
        assert mock_logits.verify_softmax_confidence(logits, 0.731, tolerance=0.01)
        
        # Test mit falschen Erwartungen
        assert not mock_logits.verify_softmax_confidence(logits, 0.9, tolerance=0.01)
        
        # Test mit leerer Liste
        assert not mock_logits.verify_softmax_confidence([], 0.5)


class TestMockClassifierWithLogits:
    """Tests für mock_classifier_with_logits Fixture"""
    
    def test_mock_classifier_fixture(self, mock_classifier_with_logits):
        """Test dass mock_classifier_with_logits Fixture funktioniert"""
        classifier, set_logits = mock_classifier_with_logits
        
        assert hasattr(classifier, 'confidence_threshold')
        assert hasattr(classifier, 'labels')
        assert hasattr(classifier, '_forward_logits')
        assert hasattr(classifier, '_label_list')
        assert callable(set_logits)
    
    def test_high_confidence_classification(self, mock_classifier_with_logits, mock_logits):
        """Test High-Confidence Klassifikation mit Mock"""
        classifier, set_logits = mock_classifier_with_logits
        
        # Setze High-Confidence Logits
        set_logits(mock_logits.high_confidence_2_classes(winner_idx=1))
        
        # Klassifiziere
        result = classifier._postprocess_logits(
            mock_logits.high_confidence_2_classes(winner_idx=1),
            ["class_0", "class_1"],
            "test text"
        )
        
        assert result.category == "class_1"
        assert result.confidence > 0.99
        assert result.is_confident is True  # Sollte über 0.7 Threshold sein
        assert result.error is None
    
    def test_low_confidence_classification(self, mock_classifier_with_logits, mock_logits):
        """Test Low-Confidence Klassifikation mit Mock"""
        classifier, set_logits = mock_classifier_with_logits
        
        # Setze Low-Confidence Logits
        set_logits(mock_logits.low_confidence_2_classes())
        
        # Klassifiziere
        result = classifier._postprocess_logits(
            mock_logits.low_confidence_2_classes(),
            ["class_0", "class_1"], 
            "test text"
        )
        
        assert result.category == "class_1"  # Höchste Wahrscheinlichkeit
        assert result.confidence < 0.6       # Unter Threshold
        assert result.is_confident is False  # Unter 0.7 Threshold
        assert result.error is None
    
    def test_threshold_boundary_cases(self, mock_classifier_with_logits):
        """Test Grenzfälle am Threshold"""
        classifier, set_logits = mock_classifier_with_logits
        
        # Test genau am Threshold (0.7)
        # Logits berechnen die genau 0.7 ergeben
        # ln(0.7/(1-0.7)) = ln(0.7/0.3) ≈ 0.847
        threshold_logits = [0.0, 0.847]
        
        result = classifier._postprocess_logits(
            threshold_logits,
            ["class_0", "class_1"],
            "boundary test"
        )
        
        # Sollte sehr nah an 0.7 sein
        assert abs(result.confidence - 0.7) < 0.01
        assert result.is_confident is True  # >= threshold


class TestIntegrationWithRealClassifier:
    """Integration Tests mit echtem Classifier (ohne Model-Loading)"""
    
    def test_confidence_threshold_from_config(self, high_confidence_threshold):
        """Test dass Classifier Threshold aus Config lädt"""
        from bu_processor.core.config import get_config
        
        config = get_config()
        threshold = config.ml_model.classifier_confidence_threshold
        
        # Sollte 0.9 durch high_confidence_threshold Fixture sein
        assert abs(threshold - 0.9) < 0.001
    
    def test_softmax_method_directly(self):
        """Test der _softmax Methode direkt"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Test mit verschiedenen Logits
        test_cases = [
            ([1.0, 2.0, 3.0], 2),           # Index 2 sollte höchste Prob haben
            ([-2.0, 6.0], 1),               # Index 1 sollte höchste Prob haben  
            ([0.0, 0.0, 0.0], None),        # Alle gleich -> jeder Index möglich
        ]
        
        for logits, expected_winner in test_cases:
            probs = RealMLClassifier._softmax(logits)
            
            # Basis-Validierungen
            assert len(probs) == len(logits)
            assert abs(sum(probs) - 1.0) < 1e-6  # Summe = 1
            assert all(p >= 0 for p in probs)     # Alle >= 0
            
            if expected_winner is not None:
                max_idx = probs.index(max(probs))
                assert max_idx == expected_winner
    
    def test_postprocess_with_different_thresholds(self, mock_logits):
        """Test _postprocess_logits mit verschiedenen Thresholds"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Mock Classifier mit verschiedenen Thresholds
        thresholds = [0.5, 0.7, 0.9]
        logits = mock_logits.medium_confidence_2_classes()  # ~0.731 Confidence
        
        for threshold in thresholds:
            classifier = RealMLClassifier.__new__(RealMLClassifier)
            classifier.confidence_threshold = threshold
            
            result = classifier._postprocess_logits(
                logits,
                ["class_0", "class_1"],
                "test"
            )
            
            expected_confident = result.confidence >= threshold
            assert result.is_confident == expected_confident, f"Threshold {threshold} failed"


def test_comprehensive_mock_workflow(mock_classifier_with_logits, mock_logits):
    """Umfassender Test des Mock-Workflows"""
    classifier, set_logits = mock_classifier_with_logits
    
    # Test 1: High Confidence
    set_logits(mock_logits.high_confidence_2_classes(winner_idx=0))
    result = classifier._postprocess_logits(
        mock_logits.high_confidence_2_classes(winner_idx=0),
        ["insurance", "other"],
        "Insurance document"
    )
    
    assert result.category == "insurance"
    assert result.confidence > 0.99
    assert result.is_confident is True
    
    # Test 2: Low Confidence  
    set_logits(mock_logits.low_confidence_2_classes())
    result = classifier._postprocess_logits(
        mock_logits.low_confidence_2_classes(),
        ["insurance", "other"],
        "Unclear document"
    )
    
    assert result.confidence < 0.6
    assert result.is_confident is False
    
    # Test 3: Metadata prüfen
    assert "all_probabilities" in result.metadata
    assert "confidence_threshold" in result.metadata
    assert "softmax_applied" in result.metadata
    assert result.metadata["softmax_applied"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
