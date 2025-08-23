"""Intent classification and routing for BU Processor."""

import os
from typing import Tuple, Optional
import structlog
from .schema import Intent, IntentConfig, INTENT_KEYWORDS

logger = structlog.get_logger("intent.router")


class IntentRouter:
    """Routes user input to appropriate processing pipeline based on intent classification"""
    
    def __init__(self, config: Optional[IntentConfig] = None):
        self.config = config or IntentConfig()
        
        # Load threshold from environment
        threshold_env = os.getenv("BU_INTENT_THRESHOLD")
        if threshold_env:
            self.config.confidence_threshold = float(threshold_env)
        
        # Initialize classifier (try to reuse existing ML classifier)
        self.classifier = None
        try:
            from ..pipeline.classifier import RealMLClassifier
            self.classifier = RealMLClassifier(lazy=False)
            logger.info("Intent classifier initialized with ML model", 
                       threshold=self.config.confidence_threshold)
        except Exception as e:
            logger.warning("ML classifier unavailable, using fallback", error=str(e))
            self.classifier = None

    def classify_intent(self, text: str) -> Tuple[str, float]:
        """
        Classify user input intent using existing ML classifier.
        Returns (intent_label, confidence_score)
        """
        try:
            if self.classifier:
                # Use existing ML classifier
                result = self.classifier.classify_text(text)
                
                # Map BU categories to intents (adjust based on your model's output)
                # This mapping may need adjustment based on your actual classifier categories
                category_to_intent = {
                    0: Intent.ADVICE,      # FAQ/General info
                    1: Intent.APPLICATION, # Application/Contract
                    2: Intent.RISK,        # Risk assessment
                    3: Intent.ADVICE,      # Default to advice
                    4: Intent.ADVICE,      # Default to advice
                }
                
                intent = category_to_intent.get(result.category, Intent.ADVICE)
                return intent.value, result.confidence
            
            else:
                # Fallback to keyword-based classification
                return self._keyword_classification(text)
                
        except Exception as e:
            logger.error("Intent classification failed", error=str(e), text=text[:100])
            return Intent.ADVICE.value, 0.0

    def _keyword_classification(self, text: str) -> Tuple[str, float]:
        """Fallback keyword-based intent classification"""
        text_lower = text.lower()
        scores = {}
        
        for intent, keywords in INTENT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[intent] = score / len(keywords)
        
        if not scores:
            return Intent.ADVICE.value, 0.1
        
        best_intent = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(scores[best_intent] * 2, 0.9)  # Scale up but cap at 0.9
        
        return best_intent.value, confidence

    def route(self, text: str, threshold: Optional[float] = None) -> str:
        """
        Route user input to appropriate intent category.
        Returns intent string.
        """
        threshold = threshold or self.config.confidence_threshold
        intent, confidence = self.classify_intent(text)
        
        logger.info("Intent classified", 
                   intent=intent, 
                   confidence=confidence,
                   threshold=threshold,
                   text_preview=text[:50])
        
        # Apply confidence threshold
        if confidence < threshold:
            if self.config.fallback_enabled:
                logger.info("Confidence below threshold, using default intent",
                           confidence=confidence, threshold=threshold)
                return self.config.default_intent.value
            else:
                return Intent.OOS.value
        
        return intent


# Global router instance
_router_instance = None


def get_intent_router() -> IntentRouter:
    """Get singleton intent router instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = IntentRouter()
    return _router_instance


def route(text: str, threshold: float = 0.80) -> str:
    """Convenience function for intent routing"""
    router = get_intent_router()
    return router.route(text, threshold)
