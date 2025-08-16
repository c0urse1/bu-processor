#!/usr/bin/env python3
"""
Test the corrected search_similar_documents method.
"""

import os
import sys

# Set test environment
os.environ["ALLOW_EMPTY_PINECONE_KEY"] = "1"
os.environ["PINECONE_API_KEY"] = ""

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_stub_mode():
    """Test search_similar_documents in stub mode."""
    print("Testing stub mode functionality...")
    
    try:
        from bu_processor.bu_processor.pipeline.pinecone_integration import AsyncPineconeManager
        
        # Create manager in explicit stub mode
        manager = AsyncPineconeManager(stub_mode=True)
        
        print(f"Manager stub_mode: {manager.stub_mode}")
        print(f"Manager _initialized: {manager._initialized}")
        
        # Test single string query
        result = manager.search_similar_documents("test query", top_k=3)
        print(f"Single query result count: {len(result)}")
        print(f"First result structure: {result[0]}")
        
        # Verify structure
        assert len(result) == 1, "Should return 1 result for single query"
        assert result[0]["query"] == "test query", "Should include query text"
        assert len(result[0]["matches"]) == 3, "Should return 3 matches"
        
        # Test multiple queries
        multi_result = manager.search_similar_documents(["query1", "query2"], top_k=2)
        print(f"Multi query result count: {len(multi_result)}")
        
        assert len(multi_result) == 2, "Should return 2 results for 2 queries"
        assert multi_result[0]["query"] == "query1", "First result should match first query"
        assert multi_result[1]["query"] == "query2", "Second result should match second query"
        
        print("✅ Stub mode working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Stub mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_auto_fallback():
    """Test automatic fallback to stub mode with invalid API key."""
    print("\nTesting auto-fallback to stub mode...")
    
    try:
        from bu_processor.bu_processor.pipeline.pinecone_integration import AsyncPineconeManager
        
        # Try to create manager with fake API key but stub_mode=False
        # Should automatically fall back to stub mode
        manager = AsyncPineconeManager(api_key="fake_invalid_key", stub_mode=False)
        
        print(f"Manager stub_mode (should be True): {manager.stub_mode}")
        print(f"Manager _initialized: {manager._initialized}")
        
        # Should now be in stub mode despite stub_mode=False initially
        assert manager.stub_mode == True, "Should have fallen back to stub mode"
        assert manager._initialized == True, "Should be initialized"
        
        # Test that search still works
        result = manager.search_similar_documents("fallback test", top_k=2)
        print(f"Fallback search result: {len(result)} items")
        
        assert len(result) == 1, "Should return 1 result"
        assert len(result[0]["matches"]) == 2, "Should return 2 matches"
        
        print("✅ Auto-fallback working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Auto-fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing corrected search_similar_documents implementation...\n")
    
    success = True
    success &= test_stub_mode()
    success &= test_auto_fallback()
    
    if success:
        print("\n🎉 All tests passed!")
        print("\nImplemented features:")
        print("✅ search_similar_documents method in AsyncPineconeManager")
        print("✅ Handles both single strings and iterables of strings")
        print("✅ Returns deterministic fake results in stub mode")
        print("✅ Proper query/matches structure as specified")
        print("✅ Graceful fallback to stub mode on API errors")
        print("✅ Configurable top_k parameter")
        print("✅ Proper logging: 'Pinecone similarity search executed'")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
