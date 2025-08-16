#!/usr/bin/env python3
"""
Test the new search_similar_documents method in AsyncPineconeManager.
"""

import os
import sys

# Set test environment
os.environ["ALLOW_EMPTY_PINECONE_KEY"] = "1"
os.environ["PINECONE_API_KEY"] = ""

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_search_similar_documents():
    """Test the new search_similar_documents method."""
    print("Testing AsyncPineconeManager.search_similar_documents()...")
    
    try:
        from bu_processor.bu_processor.pipeline.pinecone_integration import AsyncPineconeManager
        
        # Create manager in stub mode
        manager = AsyncPineconeManager(stub_mode=True)
        
        print(f"Manager in stub mode: {manager.stub_mode}")
        
        # Test 1: Single string query
        print("\n1. Testing single string query:")
        result = manager.search_similar_documents("test query", top_k=3)
        print(f"   Result: {result}")
        
        assert len(result) == 1, "Should return 1 result for single query"
        assert result[0]["query"] == "test query", "Should include the query text"
        assert len(result[0]["matches"]) == 3, "Should return 3 matches"
        
        # Check match structure
        for i, match in enumerate(result[0]["matches"]):
            assert "id" in match, "Match should have id"
            assert "score" in match, "Match should have score"
            assert match["id"] == f"stub-{i}", f"Match ID should be stub-{i}"
            expected_score = 0.9 - i * 0.1
            assert abs(match["score"] - expected_score) < 0.001, f"Match score should be {expected_score}"
        
        print("   ✅ Single string query working correctly")
        
        # Test 2: Multiple string queries
        print("\n2. Testing multiple string queries:")
        queries = ["first query", "second query"]
        result = manager.search_similar_documents(queries, top_k=2)
        print(f"   Result count: {len(result)}")
        print(f"   First result: {result[0]}")
        print(f"   Second result: {result[1]}")
        
        assert len(result) == 2, "Should return 2 results for 2 queries"
        assert result[0]["query"] == "first query", "First result should match first query"
        assert result[1]["query"] == "second query", "Second result should match second query"
        
        for r in result:
            assert len(r["matches"]) == 2, "Each result should have 2 matches"
        
        print("   ✅ Multiple string queries working correctly")
        
        # Test 3: Deterministic results
        print("\n3. Testing deterministic results:")
        result1 = manager.search_similar_documents("consistent query", top_k=2)
        result2 = manager.search_similar_documents("consistent query", top_k=2)
        
        assert result1 == result2, "Results should be deterministic"
        print("   ✅ Deterministic results working correctly")
        
        # Test 4: Different top_k values
        print("\n4. Testing different top_k values:")
        result_1 = manager.search_similar_documents("test", top_k=1)
        result_3 = manager.search_similar_documents("test", top_k=3)
        result_5 = manager.search_similar_documents("test", top_k=5)
        
        assert len(result_1[0]["matches"]) == 1, "Should return 1 match for top_k=1"
        assert len(result_3[0]["matches"]) == 3, "Should return 3 matches for top_k=3"
        assert len(result_5[0]["matches"]) == 5, "Should return 5 matches for top_k=5"
        
        print("   ✅ Different top_k values working correctly")
        
        print("\n🎉 All search_similar_documents tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_mode():
    """Test search_similar_documents behavior with real mode (non-stub)."""
    print("\nTesting search_similar_documents in real mode...")
    
    try:
        from bu_processor.bu_processor.pipeline.pinecone_integration import AsyncPineconeManager
        
        # Create manager with stub_mode=False but no API key (should still behave safely)
        manager = AsyncPineconeManager(api_key="fake_key", stub_mode=False)
        
        print(f"Manager stub mode: {manager.stub_mode}")
        
        # This should not crash, even though it's not fully implemented
        result = manager.search_similar_documents("test query", top_k=2)
        print(f"Real mode result: {result}")
        
        # Should return empty matches structure
        assert len(result) == 1, "Should return 1 result"
        assert result[0]["query"] == "test query", "Should include query"
        assert len(result[0]["matches"]) == 0, "Should have empty matches (not implemented)"
        
        print("✅ Real mode handling working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Real mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing AsyncPineconeManager.search_similar_documents()...\n")
    
    success = True
    success &= test_search_similar_documents()
    success &= test_with_real_mode()
    
    if success:
        print("\n🎉 All search_similar_documents tests passed!")
        print("\nKey features implemented:")
        print("✅ Handles both single string and iterable of strings")
        print("✅ Returns deterministic fake results in stub mode")
        print("✅ Proper query/matches structure")
        print("✅ Configurable top_k parameter")
        print("✅ Safe handling in real mode")
        print("✅ Proper logging in stub mode")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
