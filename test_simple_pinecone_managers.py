#!/usr/bin/env python3
"""Simple test for Pinecone manager improvements."""

import os
import sys

# Set test environment
os.environ["ALLOW_EMPTY_PINECONE_KEY"] = "1"
os.environ["PINECONE_API_KEY"] = ""

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from bu_processor.bu_processor.pipeline.pinecone_integration import (
        AsyncPineconeManager, 
        PineconeManager,
        get_pinecone_manager
    )
    
    print("1. Testing AsyncPineconeManager...")
    async_mgr = AsyncPineconeManager(stub_mode=True)
    print(f"   stub_mode: {async_mgr.stub_mode}")
    print(f"   _initialized: {async_mgr._initialized}")
    print(f"   _dimension: {async_mgr._dimension}")
    print("   ✅ AsyncPineconeManager OK")
    
    print("\n2. Testing PineconeManager...")
    sync_mgr = PineconeManager(stub_mode=True)
    print(f"   stub_mode: {sync_mgr.stub_mode}")
    print(f"   _initialized: {sync_mgr._initialized}")
    print(f"   _dimension: {sync_mgr._dimension}")
    
    # Test methods
    result = sync_mgr.upsert_vectors([("test", [0.1]*384, {})])
    print(f"   upsert result: {result}")
    
    search_results = sync_mgr.search_similar_documents("test", top_k=2)
    print(f"   search results: {len(search_results)} items")
    print("   ✅ PineconeManager OK")
    
    print("\n3. Testing factory...")
    factory_mgr = get_pinecone_manager()
    print(f"   factory type: {type(factory_mgr).__name__}")
    print("   ✅ Factory OK")
    
    print("\n🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
