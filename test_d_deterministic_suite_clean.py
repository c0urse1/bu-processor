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
    
    # Change to tests directory
    original_dir = os.getcwd()
    os.chdir(r"c:\ml_classifier_poc\bu_processor\tests")
    
    try:
        # Run D1: Semantic Chunker Tests
        print("Executing D1: python test_semantic_chunker_real.py")
        d1_result = os.system("python test_semantic_chunker_real.py")
        if d1_result == 0:
            total_passed += 5
            print("✅ D1 semantic chunker tests completed successfully")
        else:
            total_failed += 5
            print("❌ D1 semantic chunker tests failed")
        
        print()
        
        # Run D2: Context Packer Tests
        print("Executing D2: python test_context_packer.py") 
        d2_result = os.system("python test_context_packer.py")
        if d2_result == 0:
            total_passed += 7
            print("✅ D2 context packer tests completed successfully")
        else:
            total_failed += 7
            print("❌ D2 context packer tests failed")
        
    finally:
        # Restore original directory
        os.chdir(original_dir)
    
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
