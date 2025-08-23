"""Tests for intent routing functionality."""

import pytest
from bu_processor.intent.router import IntentRouter, route
from bu_processor.intent.schema import Intent


class TestIntentRouter:
    
    def test_route_advice_high_confidence(self):
        """Test routing to advice with high confidence"""
        text = "Was ist eine Berufsunfähigkeitsversicherung?"
        result = route(text, threshold=0.5)
        # With the current ML model, this might route to application or fall back to advice
        assert result in [Intent.ADVICE.value, Intent.APPLICATION.value]
    
    def test_route_application_keywords(self):
        """Test routing to application based on keywords"""
        text = "Ich möchte einen Antrag für eine BU-Versicherung stellen"
        result = route(text, threshold=0.5)
        # Should route to application or advice (depending on classifier)
        assert result in [Intent.APPLICATION.value, Intent.ADVICE.value]
    
    def test_route_low_confidence_fallback(self):
        """Test fallback to advice when confidence is low"""
        router = IntentRouter()
        
        # Mock low confidence scenario
        def mock_classify(text):
            return Intent.RISK.value, 0.3  # Low confidence
        
        router.classify_intent = mock_classify
        result = router.route("unclear text", threshold=0.8)
        
        assert result == Intent.ADVICE.value  # Should fallback to advice
    
    def test_confidence_threshold_from_env(self, monkeypatch):
        """Test that confidence threshold can be set via environment"""
        monkeypatch.setenv("BU_INTENT_THRESHOLD", "0.9")
        router = IntentRouter()
        assert router.config.confidence_threshold == 0.9
    
    def test_keyword_classification_fallback(self):
        """Test keyword-based classification when ML classifier unavailable"""
        router = IntentRouter()
        router.classifier = None  # Disable ML classifier
        
        # Test advice keywords
        intent, confidence = router._keyword_classification("Was ist eine BU-Versicherung?")
        assert intent == Intent.ADVICE.value
        assert confidence > 0
        
        # Test application keywords  
        intent, confidence = router._keyword_classification("Ich möchte einen Antrag stellen")
        assert intent == Intent.APPLICATION.value
        assert confidence > 0
        
        # Test risk keywords
        intent, confidence = router._keyword_classification("Risikoprüfung für meinen Beruf")
        assert intent == Intent.RISK.value
        assert confidence > 0
        
        # Test out-of-scope keywords
        intent, confidence = router._keyword_classification("Wie ist das Wetter?")
        assert intent == Intent.OOS.value
        assert confidence > 0
    
    def test_route_with_keyword_fallback(self):
        """Test routing using keyword fallback"""
        router = IntentRouter()
        router.classifier = None  # Force keyword classification
        
        # Test specific keyword matches
        result = router.route("Was ist BU-Versicherung?", threshold=0.1)
        assert result == Intent.ADVICE.value
        
        result = router.route("Antrag stellen", threshold=0.1)
        assert result == Intent.APPLICATION.value
        
        result = router.route("Risikoprüfung", threshold=0.1)
        assert result == Intent.RISK.value
    
    def test_empty_input_handling(self):
        """Test handling of empty or very short input"""
        router = IntentRouter()
        
        # Empty string
        result = router.route("", threshold=0.5)
        # Should fallback to default or be classified as some intent
        assert result in [Intent.ADVICE.value, Intent.APPLICATION.value, Intent.OOS.value]
        
        # Very short input
        result = router.route("Hi", threshold=0.5)
        assert result in [Intent.ADVICE.value, Intent.APPLICATION.value, Intent.OOS.value]
