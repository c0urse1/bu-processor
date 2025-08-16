#!/usr/bin/env python3
"""
Test script for Pinecone reliable stub mode and exports.
"""

import os
import sys
from unittest.mock import patch

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_environment_gating():
    """Test robust environment gating at module level."""
    print("=== Testing Environment Gating ===")
    
    # Test without API key, without ALLOW_EMPTY_PINECONE_KEY
    with patch.dict(os.environ, {"PINECONE_API_KEY": "", "ALLOW_EMPTY_PINECONE_KEY": "0"}, clear=True):
        # Force reload of the module to test environment gating
        if 'bu_processor.bu_processor.pipeline.pinecone_integration' in sys.modules:
            del sys.modules['bu_processor.bu_processor.pipeline.pinecone_integration']
        
        from bu_processor.bu_processor.pipeline.pinecone_integration import (
            PINECONE_AVAILABLE, STUB_MODE_DEFAULT, ALLOW_EMPTY_PINECONE_KEY
        )
        
        print(f"   No API key, ALLOW_EMPTY_PINECONE_KEY=0:")
        print(f"     PINECONE_AVAILABLE: {PINECONE_AVAILABLE}")
        print(f"     STUB_MODE_DEFAULT: {STUB_MODE_DEFAULT}")
        print(f"     ALLOW_EMPTY_PINECONE_KEY: {ALLOW_EMPTY_PINECONE_KEY}")
        
        assert not PINECONE_AVAILABLE, "Should not be available without API key"
        assert not STUB_MODE_DEFAULT, "Should not default to stub mode without ALLOW_EMPTY_PINECONE_KEY"
        assert not ALLOW_EMPTY_PINECONE_KEY, "Should be False"
    
    # Test without API key, with ALLOW_EMPTY_PINECONE_KEY
    with patch.dict(os.environ, {"PINECONE_API_KEY": "", "ALLOW_EMPTY_PINECONE_KEY": "1"}, clear=True):
        # Force reload of the module to test environment gating
        if 'bu_processor.bu_processor.pipeline.pinecone_integration' in sys.modules:
            del sys.modules['bu_processor.bu_processor.pipeline.pinecone_integration']
        
        from bu_processor.bu_processor.pipeline.pinecone_integration import (
            PINECONE_AVAILABLE, STUB_MODE_DEFAULT, ALLOW_EMPTY_PINECONE_KEY
        )
        
        print(f"   No API key, ALLOW_EMPTY_PINECONE_KEY=1:")
        print(f"     PINECONE_AVAILABLE: {PINECONE_AVAILABLE}")
        print(f"     STUB_MODE_DEFAULT: {STUB_MODE_DEFAULT}")
        print(f"     ALLOW_EMPTY_PINECONE_KEY: {ALLOW_EMPTY_PINECONE_KEY}")
        
        assert not PINECONE_AVAILABLE, "Should not be available without API key"
        assert STUB_MODE_DEFAULT, "Should default to stub mode with ALLOW_EMPTY_PINECONE_KEY"
        assert ALLOW_EMPTY_PINECONE_KEY, "Should be True"
    
    print("   ✅ Environment gating working correctly")
    print()

def test_pinecone_manager_aliases():
    """Test that reliable PineconeManager aliases are available."""
    print("=== Testing PineconeManager Aliases ===")
    
    # Set test environment
    with patch.dict(os.environ, {"PINECONE_API_KEY": "", "ALLOW_EMPTY_PINECONE_KEY": "1"}, clear=True):
        # Force reload
        if 'bu_processor.bu_processor.pipeline.pinecone_integration' in sys.modules:
            del sys.modules['bu_processor.bu_processor.pipeline.pinecone_integration']
        
        from bu_processor.bu_processor.pipeline.pinecone_integration import (
            get_pinecone_manager,
            DefaultPineconeManager,
            PineconeManagerAlias,
            PineconeManagerStub
        )
        
        print("   Available aliases:")
        print(f"     get_pinecone_manager: {get_pinecone_manager}")
        print(f"     DefaultPineconeManager: {DefaultPineconeManager}")
        print(f"     PineconeManagerAlias: {PineconeManagerAlias}")
        
        # Test that aliases work
        manager1 = get_pinecone_manager()
        manager2 = DefaultPineconeManager()
        manager3 = PineconeManagerAlias()
        
        print(f"     get_pinecone_manager() returns: {type(manager1).__name__}")
        print(f"     DefaultPineconeManager() returns: {type(manager2).__name__}")
        print(f"     PineconeManagerAlias() returns: {type(manager3).__name__}")
        
        # All should return stub in test mode
        assert isinstance(manager1, PineconeManagerStub), "Should return stub manager"
        assert isinstance(manager2, PineconeManagerStub), "Should return stub manager"
        assert isinstance(manager3, PineconeManagerStub), "Should return stub manager"
        
        print("   ✅ All aliases return correct stub manager")
    print()

def test_stub_search_results():
    """Test that stub returns predictable fake results."""
    print("=== Testing Stub Search Results ===")
    
    # Set test environment
    with patch.dict(os.environ, {"PINECONE_API_KEY": "", "ALLOW_EMPTY_PINECONE_KEY": "1"}, clear=True):
        # Force reload
        if 'bu_processor.bu_processor.pipeline.pinecone_integration' in sys.modules:
            del sys.modules['bu_processor.bu_processor.pipeline.pinecone_integration']
        
        from bu_processor.bu_processor.pipeline.pinecone_integration import get_pinecone_manager
        
        manager = get_pinecone_manager()
        
        # Test consistent results for same query
        query = "test query"
        results1 = manager.search_similar_documents(query, top_k=3)
        results2 = manager.search_similar_documents(query, top_k=3)
        
        print(f"   Query: '{query}'")
        print(f"   First call results: {len(results1)} items")
        print(f"   Second call results: {len(results2)} items")
        
        # Results should be identical for same query
        assert results1 == results2, "Results should be deterministic"
        
        # Check result structure
        assert len(results1) == 3, "Should return requested number of results"
        
        for i, result in enumerate(results1):
            print(f"     Result {i+1}: id='{result['id']}', score={result['score']:.2f}")
            assert "id" in result, "Result should have id"
            assert "score" in result, "Result should have score"
            assert "metadata" in result, "Result should have metadata"
            assert result["score"] > 0, "Score should be positive"
            assert "content_preview" in result["metadata"], "Metadata should have content_preview"
        
        # Test different top_k values
        results_5 = manager.search_similar_documents(query, top_k=5)
        assert len(results_5) == 5, "Should return 5 results"
        
        # Scores should decrease
        scores = [r["score"] for r in results_5]
        assert scores == sorted(scores, reverse=True), "Scores should decrease"
        
        print("   ✅ Stub search results are predictable and well-structured")
    print()

def test_exports():
    """Test that all expected exports are available."""
    print("=== Testing Exports ===")
    
    with patch.dict(os.environ, {"PINECONE_API_KEY": "", "ALLOW_EMPTY_PINECONE_KEY": "1"}, clear=True):
        # Force reload
        if 'bu_processor.bu_processor.pipeline.pinecone_integration' in sys.modules:
            del sys.modules['bu_processor.bu_processor.pipeline.pinecone_integration']
        
        # Import all expected exports
        from bu_processor.bu_processor.pipeline.pinecone_integration import (
            PineconeManager,
            PineconeManagerStub,
            get_pinecone_manager,
            DefaultPineconeManager,
            PineconeManagerAlias,
            PINECONE_AVAILABLE,
            PINECONE_SDK_AVAILABLE,
            STUB_MODE_DEFAULT,
            ALLOW_EMPTY_PINECONE_KEY,
        )
        
        expected_exports = [
            "PineconeManager",
            "PineconeManagerStub", 
            "get_pinecone_manager",
            "DefaultPineconeManager",
            "PineconeManagerAlias",
            "PINECONE_AVAILABLE",
            "PINECONE_SDK_AVAILABLE",
            "STUB_MODE_DEFAULT",
            "ALLOW_EMPTY_PINECONE_KEY",
        ]
        
        print("   Available exports:")
        for export in expected_exports:
            print(f"     ✅ {export}")
        
        print("   ✅ All expected exports are available")
    print()

if __name__ == "__main__":
    print("Testing Pinecone reliable stub mode and exports...\n")
    
    try:
        test_environment_gating()
        test_pinecone_manager_aliases()
        test_stub_search_results()
        test_exports()
        
        print("🎉 All Pinecone stub mode tests passed!")
        print("\nKey improvements implemented:")
        print("✅ Robust environment gating at module top")
        print("✅ Reliable PineconeManager aliases for tests")
        print("✅ Predictable fake search results in stub mode")
        print("✅ Clear logging when in stub mode")
        print("✅ Proper ALLOW_EMPTY_PINECONE_KEY=1 support")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
