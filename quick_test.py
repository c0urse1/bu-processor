#!/usr/bin/env python3
import os, sys
os.environ["ALLOW_EMPTY_PINECONE_KEY"] = "1"
os.environ["PINECONE_API_KEY"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from bu_processor.bu_processor.pipeline.pinecone_integration import AsyncPineconeManager
    print("Import successful")
    
    manager = AsyncPineconeManager(stub_mode=True)
    print(f"Manager created, stub_mode: {manager.stub_mode}")
    
    result = manager.search_similar_documents("test", top_k=2)
    print(f"Result: {result}")
    print("SUCCESS!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
