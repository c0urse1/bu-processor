#!/usr/bin/env python3
"""
Test the improved AsyncPineconeManager and PineconeManager stub mode.
"""

import os
import sys
from unittest.mock import patch

# Set test environment before importing
os.environ["ALLOW_EMPTY_PINECONE_KEY"] = "1"
os.environ["PINECONE_API_KEY"] = ""

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_async_manager_stub_mode():
    """Test AsyncPineconeManager stub mode."""
    print("=== Testing AsyncPineconeManager Stub Mode ===")
    
    try:
        from bu_processor.bu_processor.pipeline.pinecone_integration import AsyncPineconeManager
        
        # Test with explicit stub mode
        manager = AsyncPineconeManager(stub_mode=True)
        
        print(f"Manager stub_mode: {manager.stub_mode}")
        print(f"Manager _initialized: {manager._initialized}")
        print(f"Manager _dimension: {manager._dimension}")
        
        assert manager.stub_mode == True, "Should be in stub mode"
        assert manager._initialized == True, "Should be initialized"
        assert manager._dimension == 384, "Should have deterministic dimension"
        
        print("✅ AsyncPineconeManager stub mode working correctly")
        
        # Test with no API key (should auto-stub)
        manager2 = AsyncPineconeManager(api_key=None)
        print(f"Manager2 stub_mode (no API key): {manager2.stub_mode}")
        assert manager2.stub_mode == True, "Should auto-enable stub mode without API key"
        
        print("✅ AsyncPineconeManager auto-stub working correctly")
        
    except Exception as e:
        print(f"❌ AsyncPineconeManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True

def test_sync_manager_stub_mode():
    """Test PineconeManager stub mode."""
    print("=== Testing PineconeManager Stub Mode ===")
    
    try:
        from bu_processor.bu_processor.pipeline.pinecone_integration import PineconeManager
        
        # Test with explicit stub mode
        manager = PineconeManager(stub_mode=True)
        
        print(f"Manager stub_mode: {manager.stub_mode}")
        print(f"Manager _initialized: {manager._initialized}")
        print(f"Manager _dimension: {manager._dimension}")
        
        assert manager.stub_mode == True, "Should be in stub mode"
        assert manager._initialized == True, "Should be initialized"
        assert manager._dimension == 384, "Should have deterministic dimension"
        
        print("✅ PineconeManager stub mode working correctly")
        
        # Test upsert in stub mode
        fake_vectors = [("doc1", [0.1] * 384, {"title": "test"})]
        result = manager.upsert_vectors(fake_vectors)
        print(f"Upsert result: {result}")
        assert result["upserted"] == 1, "Should return fake upsert count"
        
        print("✅ PineconeManager upsert stub working correctly")
        
        # Test search in stub mode
        search_results = manager.search_similar_documents("test query", top_k=3)
        print(f"Search results: {len(search_results)} items")
        
        if search_results:
            print(f"First result: {search_results[0]}")
            assert len(search_results) == 3, "Should return requested number of results"
            assert "id" in search_results[0], "Results should have id"
            assert "score" in search_results[0], "Results should have score"
            assert "metadata" in search_results[0], "Results should have metadata"
        
        print("✅ PineconeManager search stub working correctly")
        
    except Exception as e:
        print(f"❌ PineconeManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True

def test_factory_function():
    """Test get_pinecone_manager factory with stub mode."""
    print("=== Testing get_pinecone_manager Factory ===")
    
    try:
        from bu_processor.bu_processor.pipeline.pinecone_integration import get_pinecone_manager
        
        # Should return stub manager in test environment
        manager = get_pinecone_manager()
        print(f"Factory returned: {type(manager).__name__}")
        
        # Test that it has search method
        assert hasattr(manager, 'search_similar_documents'), "Should have search method"
        
        # Test search functionality
        results = manager.search_similar_documents("factory test", top_k=2)
        print(f"Factory manager search results: {len(results)} items")
        
        assert len(results) == 2, "Should return requested number of results"
        
        print("✅ Factory function working correctly")
        
    except Exception as e:
        print(f"❌ Factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True

if __name__ == "__main__":
    print("Testing improved Pinecone manager stub mode...\n")
    
    success = True
    success &= test_async_manager_stub_mode()
    success &= test_sync_manager_stub_mode()
    success &= test_factory_function()
    
    if success:
        print("🎉 All Pinecone manager stub mode tests passed!")
        print("\nKey improvements implemented:")
        print("✅ AsyncPineconeManager honors stub_mode parameter")
        print("✅ PineconeManager honors stub_mode parameter")
        print("✅ Deterministic _dimension values in stub mode")
        print("✅ Proper _initialized flag handling")
        print("✅ Auto-stub when API key missing")
        print("✅ Predictable fake results in stub methods")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
