# scripts/test_simplified_pipeline.py
"""
Test script for the simplified Pinecone pipeline.
This demonstrates the cleaned-up MVP functionality.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bu_processor.pipeline.simplified_upsert import SimplifiedUpsertPipeline
from bu_processor.core.logging_setup import get_logger

logger = get_logger("test_simplified_pipeline")

def test_simplified_pipeline():
    """Test the simplified upsert pipeline."""
    
    print("🧪 Testing Simplified Upsert Pipeline")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        print("1. Initializing SimplifiedUpsertPipeline...")
        pipeline = SimplifiedUpsertPipeline()
        
        print(f"   ✅ Embedder initialized: {pipeline.embedder.model_name}")
        print(f"   ✅ Embedding dimension: {pipeline.embedder.dimension}")
        print(f"   ✅ Pinecone index: {pipeline.pinecone_manager.index_name}")
        print(f"   ✅ Namespace: {pipeline.namespace}")
        
        # Prepare test data
        print("\n2. Preparing test data...")
        test_chunks = [
            {
                "text": "BU leistet bei dauerhafter Berufsunfähigkeit ab 50% Grad der Berufsunfähigkeit.",
                "page": 1,
                "section": "leistungen",
                "meta": {"importance": "high", "category": "benefits"}
            },
            {
                "text": "Die Kündigungsfrist beträgt 3 Monate zum Ende des Versicherungsjahres.",
                "page": 5,
                "section": "kuendigung", 
                "meta": {"importance": "medium", "category": "termination"}
            },
            {
                "text": "Nachversicherungsgarantie gilt bei beruflichen Veränderungen ohne Gesundheitsprüfung.",
                "page": 7,
                "section": "nachversicherung",
                "meta": {"importance": "high", "category": "guarantees"}
            }
        ]
        
        # Test upsert
        print(f"\n3. Testing document upsert with {len(test_chunks)} chunks...")
        result = pipeline.upsert_document(
            doc_id="test-bu-document-001",
            doc_title="BU-Versicherung Bedingungen",
            doc_source="test_documents/bu_bedingungen.pdf",
            chunks=test_chunks,
            doc_meta={"test": True, "version": "1.0"}
        )
        
        print(f"   ✅ Document upserted successfully:")
        print(f"      - Doc ID: {result['doc_id']}")
        print(f"      - Chunks: {result['chunks_count']}")
        print(f"      - Dimension: {result['embedding_dimension']}")
        print(f"      - Ingested at: {result['ingested_at']}")
        
        # Test query
        print(f"\n4. Testing similarity search...")
        query_results = pipeline.query_similar_documents(
            query_text="Wann zahlt die BU-Versicherung?",
            top_k=3
        )
        
        print(f"   ✅ Found {len(query_results)} similar documents:")
        for i, match in enumerate(query_results.get("matches", []), 1):
            print(f"      {i}. Score: {match.get('score', 0):.3f}")
            print(f"         ID: {match.get('id', 'N/A')}")
            print(f"         Metadata: {match.get('metadata', {})}")
        
        print(f"\n🎉 All tests passed! The simplified pipeline is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.exception("Test failed")
        return False

if __name__ == "__main__":
    success = test_simplified_pipeline()
    sys.exit(0 if success else 1)
