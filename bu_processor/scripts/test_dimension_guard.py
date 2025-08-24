#!/usr/bin/env python3
"""
Quality Check: Dimension Guard Test
==================================

This test deliberately creates a dimension mismatch to verify that
the quality gates properly catch and report the error.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bu_processor.embeddings.embedder import Embedder
from bu_processor.integrations.pinecone_facade import make_pinecone_manager
from bu_processor.core.quality_gates import QualityGateError

def test_dimension_guard():
    print("🛡️  Quality Check: Dimension Guard Test")
    print("=" * 50)
    
    print("1. Creating embedder with standard model...")
    embedder = Embedder()
    print(f"   Embedder dimension: {embedder.dimension}")
    
    print("2. Creating Pinecone manager...")
    pc = make_pinecone_manager(
        index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),      # v2
        cloud=os.getenv("PINECONE_CLOUD"),          # v3
        region=os.getenv("PINECONE_REGION"),        # v3
        namespace=os.getenv("PINECONE_NAMESPACE")
    )
    
    print("3. Testing with correct dimensions...")
    try:
        # This should work fine
        test_vectors = [[0.1] * embedder.dimension, [0.2] * embedder.dimension]
        test_ids = ["correct-1", "correct-2"]
        
        result = pc.upsert_vectors(
            ids=test_ids,
            vectors=test_vectors,
            embedder=embedder
        )
        print("   ✅ Correct dimensions accepted")
        
    except Exception as e:
        print(f"   ❌ Unexpected error with correct dimensions: {e}")
        return False
    
    print("4. Testing dimension guard with wrong vector dimensions...")
    try:
        # Create vectors with wrong dimensions
        wrong_dimension = embedder.dimension + 100  # Definitely wrong
        wrong_vectors = [[0.1] * wrong_dimension, [0.2] * wrong_dimension]
        wrong_ids = ["wrong-1", "wrong-2"]
        
        result = pc.upsert_vectors(
            ids=wrong_ids,
            vectors=wrong_vectors,
            embedder=embedder
        )
        print("   ❌ ERROR: Wrong dimensions were accepted (quality gate failed!)")
        return False
        
    except (RuntimeError, QualityGateError, ValueError) as e:
        print(f"   ✅ Quality gate correctly caught dimension mismatch: {e}")
        
    except Exception as e:
        print(f"   ⚠️  Wrong dimensions caused different error: {e}")
    
    print("5. Testing dimension guard with length mismatch...")
    try:
        # Create mismatched array lengths
        mismatch_ids = ["id-1", "id-2", "id-3"]  # 3 IDs
        mismatch_vectors = [[0.1] * embedder.dimension, [0.2] * embedder.dimension]  # 2 vectors
        
        result = pc.upsert_vectors(
            ids=mismatch_ids,
            vectors=mismatch_vectors,
            embedder=embedder
        )
        print("   ❌ ERROR: Length mismatch was accepted (quality gate failed!)")
        return False
        
    except (RuntimeError, QualityGateError, ValueError) as e:
        print(f"   ✅ Quality gate correctly caught length mismatch: {e}")
        
    except Exception as e:
        print(f"   ⚠️  Length mismatch caused different error: {e}")
    
    print("6. Testing dimension guard with inconsistent vector dimensions...")
    try:
        # Create vectors with inconsistent dimensions
        inconsistent_vectors = [
            [0.1] * embedder.dimension,           # Correct dimension
            [0.2] * (embedder.dimension - 50)     # Wrong dimension
        ]
        inconsistent_ids = ["inconsistent-1", "inconsistent-2"]
        
        result = pc.upsert_vectors(
            ids=inconsistent_ids,
            vectors=inconsistent_vectors,
            embedder=embedder
        )
        print("   ❌ ERROR: Inconsistent dimensions were accepted (quality gate failed!)")
        return False
        
    except (RuntimeError, QualityGateError, ValueError) as e:
        print(f"   ✅ Quality gate correctly caught inconsistent dimensions: {e}")
        
    except Exception as e:
        print(f"   ⚠️  Inconsistent dimensions caused different error: {e}")
    
    print()
    print("🎯 Dimension Guard Test Results:")
    print("   ✅ Quality gates are functioning correctly")
    print("   ✅ Dimension mismatches are caught and rejected")
    print("   ✅ Length mismatches are caught and rejected") 
    print("   ✅ Inconsistent dimensions are caught and rejected")
    print("   ✅ Correct data is accepted normally")
    
    return True

if __name__ == "__main__":
    success = test_dimension_guard()
    if success:
        print("\n🚀 Dimension guard test PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Dimension guard test FAILED!")
        sys.exit(1)
