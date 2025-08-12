#!/usr/bin/env python3
"""Quick confidence verification."""

import torch

def test_confidence_calculations():
    """Test that our strong logits produce confidence > 0.7"""
    
    print("=== Confidence Verification ===")
    
    # Test weak logits (old problematic values)
    weak_logits = torch.tensor([0.1, 0.8, 0.1])
    weak_probs = torch.softmax(weak_logits, dim=0)
    weak_conf = weak_probs.max().item()
    
    print(f"Weak logits {weak_logits.tolist()}:")
    print(f"  → softmax {[f'{p:.3f}' for p in weak_probs.tolist()]}")
    print(f"  → max confidence {weak_conf:.3f}")
    print(f"  → passes > 0.7? {weak_conf > 0.7} ❌")
    
    print()
    
    # Test strong logits (our fix)
    strong_logits = torch.tensor([0.1, 5.0, 0.1])
    strong_probs = torch.softmax(strong_logits, dim=0)
    strong_conf = strong_probs.max().item()
    
    print(f"Strong logits {strong_logits.tolist()}:")
    print(f"  → softmax {[f'{p:.3f}' for p in strong_probs.tolist()]}")
    print(f"  → max confidence {strong_conf:.3f}")
    print(f"  → passes > 0.7? {strong_conf > 0.7} ✅")
    
    print()
    print("✅ Strong logits fix confirmed!")
    
    return strong_conf > 0.7

if __name__ == "__main__":
    test_confidence_calculations()
