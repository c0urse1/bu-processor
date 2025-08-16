#!/usr/bin/env python3
"""
Simple test for Pinecone stub mode functionality.
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

def test_stub_functionality():
    """Test basic stub functionality."""
    print("Testing Pinecone stub mode...")
    
    try:
        from bu_processor.bu_processor.pipeline.pinecone_integration import (
            get_pinecone_manager,
            STUB_MODE_DEFAULT,
            ALLOW_EMPTY_PINECONE_KEY,
            PINECONE_AVAILABLE
        )
        
        print(f"Environment settings:")
        print(f"  PINECONE_AVAILABLE: {PINECONE_AVAILABLE}")
        print(f"  STUB_MODE_DEFAULT: {STUB_MODE_DEFAULT}")
        print(f"  ALLOW_EMPTY_PINECONE_KEY: {ALLOW_EMPTY_PINECONE_KEY}")
        print()
        
        # Get manager (should be stub)
        manager = get_pinecone_manager()
        print(f"Manager type: {type(manager).__name__}")
        
        # Test search
        results = manager.search_similar_documents("test query", top_k=3)
        print(f"Search results: {len(results)} items")
        
        if results:
            print(f"First result: {results[0]}")
            
        print("✅ Stub mode working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_stub_functionality()
    sys.exit(0 if success else 1)
