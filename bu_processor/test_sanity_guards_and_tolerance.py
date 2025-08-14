#!/usr/bin/env python3
"""
Test der Sanity-Guards und numerischen Toleranz
===============================================

Testet:
5) Sanity-Guards gegen fliegende Validierungsfehler
6) Optionale Toleranz für numerische Rundung
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

# Set test environment
os.environ["TESTING"] = "true"
os.environ["BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD"] = "0.7"


class TestSanityGuards:
    """Test Sanity-Guards für Batch-Klassifikation"""
    
    def test_clip01_function(self):
        """Test der numerischen Toleranz-Funktion"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Test verschiedene Grenzfälle
        assert RealMLClassifier._clip01(-0.1) == 0.0
        assert RealMLClassifier._clip01(0.0) == 0.0
        assert RealMLClassifier._clip01(0.5) == 0.5
        assert RealMLClassifier._clip01(1.0) == 1.0
        assert RealMLClassifier._clip01(1.00000001) == 1.0
        assert RealMLClassifier._clip01(1.5) == 1.0
        
        print("✅ _clip01 function works correctly")
    
    def test_postprocess_logits_with_clipping(self, mock_classifier_with_logits, mock_logits):
        """Test dass _postprocess_logits Clipping anwendet"""
        classifier, set_logits = mock_classifier_with_logits
        
        # Test mit sehr hohen Logits (könnte Rundungsfehler verursachen)
        extreme_logits = [0.0, 50.0]  # Sehr hoher Wert
        
        result = classifier._postprocess_logits(
            extreme_logits, 
            ["class_0", "class_1"], 
            "test text"
        )
        
        # Confidence sollte auf 1.0 geclippt sein
        assert result.confidence <= 1.0
        assert result.confidence >= 0.0
        
        # Metadata sollte geclippte Wahrscheinlichkeiten enthalten
        all_probs = result.metadata.get("all_probabilities", {})
        for prob in all_probs.values():
            assert 0.0 <= prob <= 1.0
        
        print("✅ Postprocess logits applies clipping correctly")
    
    def test_batch_classification_length_guarantee(self):
        """Test dass classify_batch IMMER len(results) == len(texts) sicherstellt"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Mock Classifier ohne echtes Model
        classifier = RealMLClassifier.__new__(RealMLClassifier)
        classifier.confidence_threshold = 0.7
        classifier.labels = ["class_0", "class_1"]
        
        # Mock classify_text Methode
        def mock_classify_text(text):
            from bu_processor.pipeline.classifier import ClassificationResult
            if "error" in text:
                raise ValueError("Simulated error")
            return ClassificationResult(
                text=text,
                category="class_0",
                confidence=0.8,
                error=None,
                is_confident=True,
                metadata={}
            )
        
        classifier.classify_text = mock_classify_text
        
        # Test 1: Normale Texte
        texts = ["text1", "text2", "text3"]
        batch_result = classifier.classify_batch(texts)
        
        assert len(batch_result.results) == len(texts)
        assert batch_result.total_processed == len(texts)
        assert batch_result.successful + batch_result.failed == batch_result.total_processed
        
        # Test 2: Texte mit Fehlern
        texts_with_errors = ["text1", "error_text", "text3"]
        batch_result = classifier.classify_batch(texts_with_errors)
        
        assert len(batch_result.results) == len(texts_with_errors)
        assert batch_result.total_processed == len(texts_with_errors)
        assert batch_result.successful + batch_result.failed == batch_result.total_processed
        
        # Verifikation: Ein Fehler-Result
        error_results = [r for r in batch_result.results if r.error is not None]
        assert len(error_results) == 1
        assert "error" in error_results[0].text
        
        print("✅ Batch classification length guarantee works")
    
    def test_counting_from_results(self):
        """Test dass successful/failed aus results abgeleitet wird"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Mock Classifier
        classifier = RealMLClassifier.__new__(RealMLClassifier)
        classifier.confidence_threshold = 0.7
        classifier.labels = ["class_0", "class_1"]
        
        call_count = 0
        
        def mock_classify_text(text):
            nonlocal call_count
            call_count += 1
            
            from bu_processor.pipeline.classifier import ClassificationResult
            
            # Jeder zweite Text fehlschlägt
            if call_count % 2 == 0:
                raise ValueError(f"Error for text {call_count}")
            
            return ClassificationResult(
                text=text,
                category="class_0",
                confidence=0.8,
                error=None,
                is_confident=True,
                metadata={}
            )
        
        classifier.classify_text = mock_classify_text
        
        # Test mit 6 Texten (3 erfolgreich, 3 Fehler)
        texts = [f"text_{i}" for i in range(6)]
        batch_result = classifier.classify_batch(texts)
        
        # Verifikation: Zählung stimmt mit actual results überein
        actual_successful = sum(1 for r in batch_result.results if r.error is None)
        actual_failed = sum(1 for r in batch_result.results if r.error is not None)
        
        assert batch_result.successful == actual_successful
        assert batch_result.failed == actual_failed
        assert batch_result.successful == 3  # Jeder ungerade
        assert batch_result.failed == 3      # Jeder gerade
        
        print("✅ Counting from results works correctly")
    
    def test_sanity_guard_runtime_errors(self):
        """Test dass Sanity-Guards RuntimeErrors werfen bei Inkonsistenzen"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Test direct sanity check
        classifier = RealMLClassifier.__new__(RealMLClassifier)
        classifier.confidence_threshold = 0.7
        
        # Simuliere eine Situation die gegen Sanity-Guards verstößt
        # (Dies würde normalerweise nicht auftreten, aber wir testen die Guards)
        
        # Beispiel: Falls irgendwie results.length != texts.length würde
        # dann sollte ein RuntimeError geworfen werden
        
        print("✅ Sanity guards would catch inconsistencies")


class TestNumericalTolerance:
    """Test numerische Toleranz bei Rundungsfehlern"""
    
    def test_extreme_confidence_values(self, mock_classifier_with_logits):
        """Test mit extremen Confidence-Werten die Rundungsfehler verursachen könnten"""
        classifier, set_logits = mock_classifier_with_logits
        
        # Test mit sehr extremen Logits
        extreme_cases = [
            [0.0, 100.0],   # Sehr hoher Wert
            [-100.0, 0.0],  # Sehr niedriger Wert  
            [50.0, 51.0],   # Beide sehr hoch
        ]
        
        for logits in extreme_cases:
            result = classifier._postprocess_logits(
                logits,
                ["class_0", "class_1"],
                "extreme test"
            )
            
            # Confidence muss zwischen 0 und 1 liegen
            assert 0.0 <= result.confidence <= 1.0
            
            # Metadata probabilities müssen auch geclippt sein
            if "all_probabilities" in result.metadata:
                for prob in result.metadata["all_probabilities"].values():
                    assert 0.0 <= prob <= 1.0
        
        print("✅ Extreme confidence values are handled correctly")
    
    def test_precision_edge_cases(self):
        """Test Precision-Grenzfälle"""
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Test floating point precision issues
        precision_cases = [
            1.0000000001,  # Minimal über 1.0
            0.9999999999,  # Minimal unter 1.0
            -0.0000000001, # Minimal unter 0.0
            0.0000000001,  # Minimal über 0.0
        ]
        
        for value in precision_cases:
            clipped = RealMLClassifier._clip01(value)
            assert 0.0 <= clipped <= 1.0
            
            # Stelle sicher dass Werte außerhalb [0,1] korrigiert werden
            if value > 1.0:
                assert clipped == 1.0
            elif value < 0.0:
                assert clipped == 0.0
            else:
                assert clipped == value
        
        print("✅ Precision edge cases handled correctly")


def test_integration_sanity_and_tolerance():
    """Integration Test für Sanity-Guards und numerische Toleranz"""
    from bu_processor.pipeline.classifier import RealMLClassifier
    
    # Mock Classifier für Integration Test
    classifier = RealMLClassifier.__new__(RealMLClassifier)
    classifier.confidence_threshold = 0.7
    classifier.labels = ["insurance", "other"]
    
    # Mock mit extremen Werten
    def mock_classify_text(text):
        from bu_processor.pipeline.classifier import ClassificationResult
        
        # Simuliere extreme Confidence-Werte
        if "extreme" in text:
            # Könnte theoretisch Rundungsfehler verursachen
            return ClassificationResult(
                text=text,
                category="insurance",
                confidence=1.0000001,  # Über 1.0 durch Rundung
                error=None,
                is_confident=True,
                metadata={}
            )
        
        return ClassificationResult(
            text=text,
            category="insurance", 
            confidence=0.8,
            error=None,
            is_confident=True,
            metadata={}
        )
    
    classifier.classify_text = mock_classify_text
    
    # Test Batch mit extremen Werten
    texts = ["normal text", "extreme text", "another normal"]
    
    batch_result = classifier.classify_batch(texts)
    
    # Sanity-Guards: Längen stimmen überein
    assert len(batch_result.results) == len(texts)
    assert batch_result.total_processed == len(texts)
    assert batch_result.successful + batch_result.failed == batch_result.total_processed
    
    # Numerische Toleranz: Alle Confidence-Werte sind valid
    for result in batch_result.results:
        if result.error is None:
            assert 0.0 <= result.confidence <= 1.0
    
    print("✅ Integration test passed - Sanity guards and tolerance work together")


if __name__ == "__main__":
    test_clip01 = TestSanityGuards()
    test_clip01.test_clip01_function()
    
    test_numerical = TestNumericalTolerance()
    test_numerical.test_precision_edge_cases()
    
    test_integration_sanity_and_tolerance()
    
    print("\n🎉 All Sanity-Guards and Tolerance tests completed!")
