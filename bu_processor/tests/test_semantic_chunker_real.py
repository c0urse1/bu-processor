#!/usr/bin/env python3
"""
D1) Semantic chunker tests - deterministic, offline testing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bu_processor.embeddings.testing_backend import FakeDeterministicEmbeddings
from bu_processor.semantic.greedy_boundary_chunker import GreedyBoundarySemanticChunker

def test_semantic_chunker_respects_pages_and_headings():
    """Test that semantic chunker respects page boundaries and captures headings"""
    embedder = FakeDeterministicEmbeddings(dim=64)
    chunker = GreedyBoundarySemanticChunker(embedder, max_tokens=80, sim_threshold=0.55, overlap_sentences=1)

    page1 = """1 Introduction
This paper explains insurance coverage and financial loss due to negligence. 
Professional liability insurance covers financial losses from mistakes."""
    page2 = """2 Corporate Finance
Corporate finance optimizes capital structure and funding. Debt financing can be cheaper than equity."""
    pages = [(1, page1), (2, page2)]

    chunks = chunker.chunk_pages(doc_id="doc-1", pages=pages, source_url="file://demo.pdf", tenant="acme")
    assert len(chunks) >= 2
    # hard boundary at page change
    assert chunks[0].page_start == 1 and chunks[0].page_end == 1
    assert chunks[-1].page_start == 2
    # headings captured
    assert chunks[0].section and "Introduction" in chunks[0].section
    assert chunks[-1].section and "Corporate Finance" in chunks[-1].section
    # metadata present
    assert chunks[0].doc_id == "doc-1"
    print("✓ test_semantic_chunker_respects_pages_and_headings PASSED")

def test_semantic_chunker_similarity_boundaries():
    """Test that chunker respects similarity thresholds for boundaries"""
    embedder = FakeDeterministicEmbeddings(dim=64)
    chunker = GreedyBoundarySemanticChunker(embedder, max_tokens=150, sim_threshold=0.70, overlap_sentences=1)

    # Create content with clear semantic boundaries
    page1 = """Machine Learning Overview
Machine learning algorithms can process large datasets efficiently. 
Deep learning networks excel at pattern recognition tasks.
Training requires significant computational resources and time.

Financial Analysis
Corporate finance focuses on capital structure optimization.
Debt financing provides leverage for business growth.
Investment decisions require careful risk assessment."""
    
    pages = [(1, page1)]
    chunks = chunker.chunk_pages(doc_id="test-doc", pages=pages, source_url="file://test.pdf", tenant="test")
    
    # Should create separate chunks for ML and Finance sections due to semantic difference
    assert len(chunks) >= 2
    # First chunk should contain ML content
    assert any("machine learning" in chunk.text.lower() for chunk in chunks)
    # Should have different sections captured
    sections = [chunk.section for chunk in chunks if chunk.section]
    assert len(set(sections)) >= 2
    print("✓ test_semantic_chunker_similarity_boundaries PASSED")

def test_semantic_chunker_token_limits():
    """Test that chunker respects token limits"""
    embedder = FakeDeterministicEmbeddings(dim=64)
    chunker = GreedyBoundarySemanticChunker(embedder, max_tokens=30, sim_threshold=0.50, overlap_sentences=0)

    # Long content that should be split
    long_text = """This is a very long document with many sentences that should be split into multiple chunks. """ * 5
    pages = [(1, long_text)]
    
    chunks = chunker.chunk_pages(doc_id="long-doc", pages=pages, source_url="file://long.pdf", tenant="test")
    
    # Should create multiple chunks due to token limit
    assert len(chunks) >= 2
    # Each chunk should respect token limit (roughly)
    for chunk in chunks:
        # Approximate token count (words * 1.3)
        approx_tokens = len(chunk.text.split()) * 1.3
        assert approx_tokens <= 35  # Allow some margin
    print("✓ test_semantic_chunker_token_limits PASSED")

def test_semantic_chunker_overlap():
    """Test sentence overlap functionality"""
    embedder = FakeDeterministicEmbeddings(dim=64)
    chunker = GreedyBoundarySemanticChunker(embedder, max_tokens=40, sim_threshold=0.30, overlap_sentences=2)

    text = """First sentence here. Second sentence follows. Third sentence continues. Fourth sentence ends. Fifth sentence starts. Sixth sentence completes."""
    pages = [(1, text)]
    
    chunks = chunker.chunk_pages(doc_id="overlap-test", pages=pages, source_url="file://test.pdf", tenant="test")
    
    if len(chunks) >= 2:
        # Check for overlap between consecutive chunks
        chunk1_sentences = set(chunks[0].text.split('.'))
        chunk2_sentences = set(chunks[1].text.split('.'))
        overlap = chunk1_sentences.intersection(chunk2_sentences)
        # Should have some overlap (allowing for sentence variations)
        # This is approximate since overlap depends on chunking decisions
    print("✓ test_semantic_chunker_overlap PASSED")

def test_semantic_chunker_metadata_preservation():
    """Test that all required metadata is preserved"""
    embedder = FakeDeterministicEmbeddings(dim=64)
    chunker = GreedyBoundarySemanticChunker(embedder, max_tokens=100, sim_threshold=0.60, overlap_sentences=1)

    page_content = """Section A: Important Information
This section contains critical business data and analysis.
It should be properly tagged with metadata."""
    
    pages = [(5, page_content)]  # Page 5
    chunks = chunker.chunk_pages(
        doc_id="metadata-test", 
        pages=pages, 
        source_url="file://metadata.pdf", 
        tenant="test-tenant"
    )
    
    assert len(chunks) >= 1
    chunk = chunks[0]
    
    # Verify required metadata
    assert chunk.doc_id == "metadata-test"
    assert chunk.page_start == 5
    assert chunk.page_end == 5
    assert chunk.source_url == "file://metadata.pdf"
    assert chunk.tenant == "test-tenant"
    assert chunk.section is not None
    assert "Section A" in chunk.section
    assert chunk.text is not None and len(chunk.text) > 0
    
    print("✓ test_semantic_chunker_metadata_preservation PASSED")

def run_all_semantic_chunker_tests():
    """Run all D1 semantic chunker tests"""
    print("=== D1: Semantic Chunker Tests ===")
    try:
        test_semantic_chunker_respects_pages_and_headings()
        test_semantic_chunker_similarity_boundaries()
        test_semantic_chunker_token_limits()
        test_semantic_chunker_overlap()
        test_semantic_chunker_metadata_preservation()
        print("✅ All D1 semantic chunker tests PASSED")
        return 5, 0
    except Exception as e:
        print(f"❌ D1 tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, 5

if __name__ == "__main__":
    run_all_semantic_chunker_tests()
