#!/usr/bin/env python3
"""Simple test runner for semantic chunking"""

import sys
sys.path.insert(0, '.')

from tests.test_semantic_chunking import (
    test_semantic_chunking_splits_by_topic_and_budget,
    test_semantic_chunking_respects_token_budget,
    test_semantic_chunking_empty_input,
    test_fake_deterministic_embeddings_consistency,
    test_fake_sentence_transformer_mock,
    test_semantic_chunking_with_sentence_transformer_mock,
    test_mock_consistency_across_calls,
    test_pytorch_style_mock_operations
)

def run_test(test_func, test_name):
    print(f"\n🧪 Running {test_name}...")
    try:
        test_func()
        print(f"✓ {test_name} PASSED!")
        return True
    except Exception as e:
        print(f"✗ {test_name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Running Semantic Chunking Tests")
    print("=" * 50)
    
    tests = [
        (test_semantic_chunking_splits_by_topic_and_budget, "Splits by topic and budget"),
        (test_semantic_chunking_respects_token_budget, "Respects token budget"),
        (test_semantic_chunking_empty_input, "Empty input handling"),
        (test_fake_deterministic_embeddings_consistency, "Deterministic embeddings consistency"),
        (test_fake_sentence_transformer_mock, "SentenceTransformer mock functionality"),
        (test_semantic_chunking_with_sentence_transformer_mock, "Semantic chunking with ST mock"),
        (test_mock_consistency_across_calls, "Mock consistency across calls"),
        (test_pytorch_style_mock_operations, "PyTorch-style mock operations (Step 6 fix)"),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All semantic chunking tests PASSED!")
        return 0
    else:
        print("❌ Some tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
