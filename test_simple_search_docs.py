#!/usr/bin/env python3
"""Simple test for search_similar_documents method."""

import os
import sys

# Set environment
os.environ["ALLOW_EMPTY_PINECONE_KEY"] = "1"
os.environ["PINECONE_API_KEY"] = ""

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from bu_processor.bu_processor.pipeline.pinecone_integration import AsyncPineconeManager
    
    print("Testing search_similar_documents...")
    
    # Create manager in stub mode
    manager = AsyncPineconeManager(stub_mode=True)
    print(f"Manager stub_mode: {manager.stub_mode}")
    
    # Test single query
    result = manager.search_similar_documents("test query", top_k=2)
    print(f"Single query result: {len(result)} items")
    print(f"First result: {result[0]}")
    
    # Test multiple queries
    result_multi = manager.search_similar_documents(["query1", "query2"], top_k=2)
    print(f"Multi query result: {len(result_multi)} items")
    
    print("✅ search_similar_documents working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
