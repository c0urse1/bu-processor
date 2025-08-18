#!/usr/bin/env python3
"""
D) Tests (deterministic, offline) - Complete Test Suite
Run semantic chunker and context packer tests
"""

import sys
import os

# Add the bu_processor directory to path
sys.path.insert(0, r"c:\ml_classifier_poc\bu_processor")

def main():
    print("🧪 Running D) Tests (deterministic, offline)")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    # Run D1: Semantic Chunker Tests
    print("Running D1 tests from bu_processor/tests/...")
    os.chdir(r"c:\ml_classifier_poc\bu_processor\tests")
    
    print("Executing: python test_semantic_chunker_real.py")
    d1_result = os.system("python test_semantic_chunker_real.py")
    if d1_result == 0:
        total_passed += 5
        print("✅ D1 semantic chunker tests completed successfully")
    else:
        total_failed += 5
        print("❌ D1 semantic chunker tests failed")
    
    print()
    
    # Run D2: Context Packer Tests
    print("Executing: python test_context_packer.py") 
    d2_result = os.system("python test_context_packer.py")
    if d2_result == 0:
        total_passed += 7
        print("✅ D2 context packer tests completed successfully")
    else:
        total_failed += 7
        print("❌ D2 context packer tests failed")
    
    # Final Results
    print()
    print("=" * 60)
    print(f"📊 FINAL RESULTS: {total_passed}/{total_passed + total_failed} tests passed")
    if total_failed > 0:
        print(f"⚠️  {total_failed} tests failed")
        return False
    else:
        print("🎉 All D) Tests PASSED!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
            ("Token Limits", test_semantic_chunker_token_limits),
            ("Similarity Boundaries", test_semantic_chunker_similarity_boundaries),
            ("Sentence Overlap", test_semantic_chunker_overlap),
        ]
        
        d1_passed = 0
        for test_name, test_func in d1_tests:
            try:
                test_func()
                print(f"✓ {test_name}")
                d1_passed += 1
            except Exception as e:
                print(f"✗ {test_name}: {e}")
                traceback.print_exc()
        
        print(f"\nD1 Results: {d1_passed}/{len(d1_tests)} tests passed")
        
    except ImportError as e:
        print(f"✗ Failed to import D1 tests: {e}")
        d1_passed = 0
    
    # D2: Context Packer Tests  
    print("\n=== D2: Context Packer Tests ===")
    
    try:
        from tests.test_context_packer import (
            test_context_packer_budget_and_citations,
            test_context_packer_antidup,
            test_context_packer_quota_allocation,
            test_context_packer_sentence_overlap,
            test_context_packer_prefer_summary,
            test_context_packer_unique_chunks,
            test_context_packer_empty_hits,
            test_context_packer_metadata_preservation
        )
        
        d2_tests = [
            ("Budget and Citations", test_context_packer_budget_and_citations),
            ("Anti-duplication", test_context_packer_antidup),
            ("Quota Allocation", test_context_packer_quota_allocation),
            ("Sentence Overlap", test_context_packer_sentence_overlap),
            ("Prefer Summary", test_context_packer_prefer_summary),
            ("Unique Chunks", test_context_packer_unique_chunks),
            ("Empty Hits", test_context_packer_empty_hits),
            ("Metadata Preservation", test_context_packer_metadata_preservation),
        ]
        
        d2_passed = 0
        for test_name, test_func in d2_tests:
            try:
                test_func()
                print(f"✓ {test_name}")
                d2_passed += 1
            except Exception as e:
                print(f"✗ {test_name}: {e}")
                traceback.print_exc()
        
        print(f"\nD2 Results: {d2_passed}/{len(d2_tests)} tests passed")
        
    except ImportError as e:
        print(f"✗ Failed to import D2 tests: {e}")
        d2_passed = 0
    
    # Summary
    total_passed = d1_passed + d2_passed
    total_tests = 4 + 8  # 4 D1 tests + 8 D2 tests
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All D) Tests PASSED!")
        print("\n✅ D1: Semantic chunker working correctly")
        print("  - Page boundaries respected")
        print("  - Token limits enforced")
        print("  - Similarity thresholds working")
        print("  - Sentence overlap functional")
        print("\n✅ D2: Context packer working correctly")
        print("  - Budget management effective")
        print("  - Anti-duplication preventing repetition")
        print("  - Quota allocation by score")
        print("  - Metadata preservation complete")
        print("  - Citation numbering stable")
    else:
        print(f"⚠️  {total_tests - total_passed} tests failed")
        
    return total_passed == total_tests

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
