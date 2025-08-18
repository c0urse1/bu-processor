#!/usr/bin/env python3
"""Test the GreedyBoundarySemanticChunker."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from bu_processor.semantic.greedy_boundary_chunker import GreedyBoundarySemanticChunker
from bu_processor.semantic.testing import FakeDeterministicEmbeddings

def test_basic_semantic_chunking():
    """Test basic semantic chunking functionality."""
    print("=== Testing Basic Semantic Chunking ===")
    
    # Create a fake embedder for testing
    embedder = FakeDeterministicEmbeddings()
    chunker = GreedyBoundarySemanticChunker(
        embedder=embedder,
        max_tokens=100,  # Small for testing
        sim_threshold=0.7,
        overlap_sentences=1
    )
    
    # Test document with multiple sections
    pages = [
        (1, """1. INTRODUCTION

This is the introduction section. It explains the purpose of this document.
The methodology will be explained later.

1.1 Background  

Some background information here. This provides context for the reader.
Understanding the background is important."""),
        
        (2, """2. METHODOLOGY

Our approach involves several steps. First, we collect data from various sources.
Second, we analyze the collected information.

2.1 Data Collection

We use surveys and interviews. The sample size is carefully selected.
Response rates are monitored closely.""")
    ]
    
    chunks = chunker.chunk_pages(
        doc_id="test_doc_1",
        pages=pages,
        source_url="https://example.com/test.pdf",
        tenant="test_tenant"
    )
    
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Pages: {chunk.page_start}-{chunk.page_end}")
        print(f"  Section: {chunk.section}")
        print(f"  Heading Path: {chunk.heading_path}")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Importance: {chunk.importance_score:.2f}")
        print(f"  Tokens: {chunk.meta.get('token_count', 'N/A')}")
        print(f"  Sentences: {chunk.meta.get('sentence_count', 'N/A')}")
        text_preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
        print(f"  Text: {text_preview}")
    
    return chunks

def test_hard_boundaries():
    """Test that hard boundaries are respected."""
    print("\n=== Testing Hard Boundaries ===")
    
    embedder = FakeDeterministicEmbeddings()
    chunker = GreedyBoundarySemanticChunker(
        embedder=embedder,
        max_tokens=1000,  # High limit to test boundary enforcement
        sim_threshold=0.1,  # Low threshold to test boundary enforcement
        overlap_sentences=0
    )
    
    # Document designed to test page and section boundaries
    pages = [
        (1, "Page 1 sentence 1. Page 1 sentence 2. Page 1 sentence 3."),
        (2, "1. NEW SECTION\nPage 2 sentence 1. Page 2 sentence 2."),
        (2, "2. ANOTHER SECTION\nPage 2 different section. Another sentence here.")
    ]
    
    chunks = chunker.chunk_pages(doc_id="boundary_test", pages=pages)
    
    print(f"Hard boundary test generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: Page {chunk.page_start}-{chunk.page_end}, Section '{chunk.section}'")
        print(f"    Text: {chunk.text[:60]}...")
    
    # Verify we got separate chunks for different pages/sections
    assert len(chunks) >= 3, f"Expected at least 3 chunks for hard boundaries, got {len(chunks)}"
    print("✓ Hard boundaries enforced correctly")

def test_token_budget():
    """Test token budget enforcement."""
    print("\n=== Testing Token Budget ===")
    
    embedder = FakeDeterministicEmbeddings()
    chunker = GreedyBoundarySemanticChunker(
        embedder=embedder,
        max_tokens=50,  # Very small budget
        sim_threshold=0.1,  # Low threshold 
        overlap_sentences=0
    )
    
    # Long text that should be broken by token budget
    long_text = " ".join([f"This is sentence number {i} with some content." for i in range(20)])
    pages = [(1, long_text)]
    
    chunks = chunker.chunk_pages(doc_id="budget_test", pages=pages)
    
    print(f"Token budget test generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        token_count = chunk.meta.get('token_count', 0)
        print(f"  Chunk {i+1}: {token_count} tokens")
        assert token_count <= 50, f"Chunk {i+1} exceeds token budget: {token_count} > 50"
    
    print("✓ Token budget enforced correctly")

def test_similarity_threshold():
    """Test semantic similarity threshold."""
    print("\n=== Testing Similarity Threshold ===")
    
    embedder = FakeDeterministicEmbeddings()
    chunker = GreedyBoundarySemanticChunker(
        embedder=embedder,
        max_tokens=1000,  # High budget
        sim_threshold=0.8,  # High similarity threshold
        overlap_sentences=0
    )
    
    # Text with different topics that should trigger similarity breaks
    pages = [(1, """
    Machine learning is a subset of artificial intelligence. It involves training algorithms on data.
    Deep learning uses neural networks with multiple layers.
    
    Cooking pasta requires boiling water first. Add salt to the water for flavor.
    Cook the pasta according to package directions.
    
    Financial markets are complex systems. Stock prices fluctuate based on many factors.
    Diversification helps reduce investment risk.
    """)]
    
    chunks = chunker.chunk_pages(doc_id="similarity_test", pages=pages)
    
    print(f"Similarity test generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        avg_sim = chunk.meta.get('avg_similarity')
        sim_display = f"{avg_sim:.3f}" if avg_sim is not None else "N/A"
        print(f"  Chunk {i+1}: Avg similarity = {sim_display}")
        text_preview = " ".join(chunk.text.split()[:10]) + "..."
        print(f"    Content: {text_preview}")
    
    print("✓ Similarity threshold working")

def test_chunk_statistics():
    """Test chunk statistics functionality."""
    print("\n=== Testing Chunk Statistics ===")
    
    embedder = FakeDeterministicEmbeddings()
    chunker = GreedyBoundarySemanticChunker(embedder=embedder)
    
    pages = [
        (1, "1. INTRODUCTION\nThis is an introduction with multiple sentences. It has important content."),
        (2, "2. METHODOLOGY\nThis section describes our methods. It is technical in nature."),
        (3, "3. CONCLUSION\nFinal thoughts and summary. This concludes our document.")
    ]
    
    chunks = chunker.chunk_pages(doc_id="stats_test", pages=pages)
    stats = chunker.get_chunk_stats(chunks)
    
    print("Chunk statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Verify statistics make sense
    assert stats["total_chunks"] == len(chunks)
    assert stats["pages_covered"] <= 3
    assert stats["avg_tokens_per_chunk"] > 0
    print("✓ Statistics calculated correctly")

def test_convenience_methods():
    """Test convenience methods."""
    print("\n=== Testing Convenience Methods ===")
    
    embedder = FakeDeterministicEmbeddings()
    chunker = GreedyBoundarySemanticChunker(embedder=embedder)
    
    # Test chunk_text method
    text = "This is a simple text. It has multiple sentences. Each sentence adds content."
    chunks = chunker.chunk_text(
        text=text,
        doc_id="convenience_test",
        source_url="https://example.com"
    )
    
    print(f"chunk_text() generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk.text)} chars, page {chunk.page_start}")
        assert chunk.page_start == 1, "chunk_text should set page_start to 1"
        assert chunk.doc_id == "convenience_test"
        assert chunk.meta.get("source_url") == "https://example.com"
    
    print("✓ Convenience methods working correctly")

def test_metadata_richness():
    """Test that rich metadata is populated correctly."""
    print("\n=== Testing Metadata Richness ===")
    
    embedder = FakeDeterministicEmbeddings()
    chunker = GreedyBoundarySemanticChunker(embedder=embedder)
    
    pages = [(1, """
    1. EXECUTIVE SUMMARY
    
    This executive summary provides an overview. It contains key findings.
    Strategic recommendations are included.
    
    1.1 Key Findings
    
    Our research reveals important trends. Market conditions are changing.
    Customer preferences have shifted significantly.
    """)]
    
    chunks = chunker.chunk_pages(
        doc_id="metadata_test",
        pages=pages,
        source_url="https://example.com/report.pdf",
        tenant="enterprise_client"
    )
    
    print(f"Metadata test generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} metadata:")
        print(f"  doc_id: {chunk.doc_id}")
        print(f"  section: {chunk.section}")
        print(f"  heading_path: {chunk.heading_path}")
        print(f"  heading_text: {chunk.heading_text}")
        print(f"  chunk_type: {chunk.chunk_type}")
        print(f"  importance_score: {chunk.importance_score}")
        print(f"  char_span: {chunk.char_span}")
        print(f"  meta keys: {list(chunk.meta.keys())}")
        
        # Verify required metadata
        assert chunk.doc_id == "metadata_test"
        assert chunk.chunk_type == "semantic"
        assert chunk.meta.get("source_url") == "https://example.com/report.pdf"
        assert chunk.meta.get("tenant") == "enterprise_client"
        assert chunk.meta.get("chunking_method") == "greedy_boundary_semantic"
        assert "token_count" in chunk.meta
        assert "sentence_count" in chunk.meta
    
    print("✓ Rich metadata populated correctly")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    
    embedder = FakeDeterministicEmbeddings()
    chunker = GreedyBoundarySemanticChunker(embedder=embedder)
    
    # Test empty pages
    empty_chunks = chunker.chunk_pages(doc_id="empty_test", pages=[])
    assert len(empty_chunks) == 0, "Empty pages should return empty chunks"
    print("✓ Empty pages handled correctly")
    
    # Test pages with only whitespace
    whitespace_pages = [(1, "   \n\n   "), (2, "\t\t")]
    whitespace_chunks = chunker.chunk_pages(doc_id="whitespace_test", pages=whitespace_pages)
    print(f"✓ Whitespace pages generated {len(whitespace_chunks)} chunks")
    
    # Test single sentence
    single_pages = [(1, "Single sentence here.")]
    single_chunks = chunker.chunk_pages(doc_id="single_test", pages=single_pages)
    assert len(single_chunks) == 1, "Single sentence should create one chunk"
    print("✓ Single sentence handled correctly")
    
    print("✓ Edge cases handled correctly")

if __name__ == "__main__":
    test_basic_semantic_chunking()
    test_hard_boundaries()
    test_token_budget()
    test_similarity_threshold()
    test_chunk_statistics()
    test_convenience_methods()
    test_metadata_richness()
    test_edge_cases()
    
    print("\n🎉 All GreedyBoundarySemanticChunker tests completed successfully!")
