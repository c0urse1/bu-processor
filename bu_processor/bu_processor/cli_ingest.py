#!/usr/bin/env python3
"""
CLI for document ingestion using the simplified pipeline.
"""

import os
from bu_processor.embeddings.embedder import Embedder
from bu_processor.integrations.pinecone_facade import make_pinecone_manager
from bu_processor.storage.sqlite_store import SQLiteStore
from bu_processor.pipeline.simplified_upsert import SimplifiedUpsertPipeline

def ingest_document(doc_title, doc_source, chunks, doc_id=None, namespace=None):
    """
    Ingest a document using the new simplified pipeline.
    
    Args:
        doc_title: Document title
        doc_source: Document source/path
        chunks: List of text chunks with metadata
        doc_id: Optional document ID
        namespace: Optional Pinecone namespace
    
    Returns:
        Dict with ingestion results
    """
    # Initialize the simplified pipeline
    pipeline = SimplifiedUpsertPipeline(namespace=namespace)
    
    # Use the new simplified API
    return pipeline.upsert_document(
        doc_id=doc_id,
        doc_title=doc_title,
        doc_source=doc_source,
        chunks=chunks,
        namespace=namespace
    )

# Legacy compatibility function
def ingest_document_legacy(doc_title, doc_source, chunks):
    """Legacy function for backward compatibility."""
    return ingest_document(doc_title, doc_source, chunks)

if __name__ == "__main__":
    # Example usage
    test_chunks = [
        {
            "text": "This is a test document chunk for ingestion.",
            "page": 1,
            "section": "introduction",
            "meta": {"importance": "high"}
        }
    ]
    
    result = ingest_document(
        doc_title="Test Document",
        doc_source="test.pdf",
        chunks=test_chunks
    )
    
    print(f"Ingestion result: {result}")
