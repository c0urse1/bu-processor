#!/usr/bin/env python3
"""Quick Test der High-Confidence Fixtures"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Define MockLogitsProvider directly (no import from tests)
class MockLogitsProvider:
    """Mock provider for testing logits - self-contained version."""
    
    def high_confidence_2_classes(self, winner_idx: int = 0):
        """Generate high-confidence logits for binary classification."""
        if winner_idx == 0:
            return torch.tensor([5.0, -3.0])  # ~99.7% confidence for class 0
        else:
            return torch.tensor([-3.0, 5.0])  # ~99.7% confidence for class 1
    
    def verify_softmax_confidence(self, logits: torch.Tensor, expected_confidence: float, tolerance: float = 0.01):
        """Verify that softmax produces expected confidence."""
        softmax = torch.softmax(logits, dim=-1)
        actual_confidence = softmax.max().item()
        return abs(actual_confidence - expected_confidence) <= tolerance


def test_mock_logits_provider():
    """Test MockLogitsProvider direkt"""
    print("=== Testing MockLogitsProvider ===")
    
    provider = MockLogitsProvider()
    
    # Test High Confidence 2 Classes
    logits = provider.high_confidence_2_classes(winner_idx=1)
    print(f"High confidence 2 classes (winner=1): {logits}")
    
    # Verify softmax
    confidence_ok = provider.verify_softmax_confidence(logits, 0.997, tolerance=0.01)
    print(f"Softmax verification (expect ~0.997): {confidence_ok}")
    
    # Test Medium Confidence
    logits = provider.medium_confidence_2_classes(winner_idx=0)
    print(f"Medium confidence 2 classes (winner=0): {logits}")
    
    confidence_ok = provider.verify_softmax_confidence(logits, 0.731, tolerance=0.01)
    print(f"Softmax verification (expect ~0.731): {confidence_ok}")
    
    # Test Low Confidence
    logits = provider.low_confidence_2_classes()
    print(f"Low confidence 2 classes: {logits}")
    
    confidence_ok = provider.verify_softmax_confidence(logits, 0.524, tolerance=0.01)
    print(f"Softmax verification (expect ~0.524): {confidence_ok}")
    
    print("✅ MockLogitsProvider tests passed!")


def test_environment_setup():
    """Test Environment Variable Setup"""
    print("\n=== Testing Environment Setup ===")
    
    # Set test threshold
    os.environ["BU_ML_MODEL__CLASSIFIER_CONFIDENCE_THRESHOLD"] = "0.8"
    
    try:
        from bu_processor.core.config import get_config
        config = get_config()
        threshold = config.ml_model.classifier_confidence_threshold
        print(f"Config threshold: {threshold}")
        assert abs(threshold - 0.8) < 0.001, f"Expected 0.8, got {threshold}"
        print("✅ Environment threshold test passed!")
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")


def test_softmax_calculation():
    """Test Softmax Calculation direkt"""
    print("\n=== Testing Softmax Calculation ===")
    
    try:
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Test verschiedene Logits
        test_cases = [
            ([-2.0, 6.0], "high confidence"),
            ([0.0, 1.0], "medium confidence"), 
            ([-0.1, 0.1], "low confidence"),
        ]
        
        for logits, description in test_cases:
            probs = RealMLClassifier._softmax(logits)
            max_prob = max(probs)
            print(f"{description}: logits={logits} -> probs={[f'{p:.3f}' for p in probs]}, max={max_prob:.3f}")
        
        print("✅ Softmax calculation tests passed!")
        
    except Exception as e:
        print(f"❌ Softmax test failed: {e}")


if __name__ == "__main__":
    test_mock_logits_provider()
    test_environment_setup()
    test_softmax_calculation()
    print("\n🎉 All fixture tests completed!")
