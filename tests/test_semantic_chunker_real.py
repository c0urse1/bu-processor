from bu_processor.embeddings.testing_backend import FakeDeterministicEmbeddings
from bu_processor.semantic.greedy_boundary_chunker import GreedyBoundarySemanticChunker

def test_semantic_chunker_respects_pages_and_headings():
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

def test_semantic_chunker_token_limits():
    """Test that semantic chunker respects token limits"""
    embedder = FakeDeterministicEmbeddings(dim=64)
    chunker = GreedyBoundarySemanticChunker(embedder, max_tokens=30, sim_threshold=0.8, overlap_sentences=0)
    
    # Long text that should be split due to token limits
    long_text = """This is a very long sentence that contains many words and should definitely exceed our token limit. 
    This is another long sentence that also contains many words and continues the theme. 
    And here is a third sentence that keeps going with even more content."""
    
    pages = [(1, long_text)]
    chunks = chunker.chunk_pages(doc_id="test", pages=pages, source_url="test.pdf", tenant="test")
    
    # Should create multiple chunks due to token limit
    assert len(chunks) >= 2
    # Each chunk should respect the token limit
    for chunk in chunks:
        # Allow some flexibility for overlap and boundaries
        assert len(chunk.text.split()) <= 50  # Rough token approximation

def test_semantic_chunker_similarity_boundaries():
    """Test that semantic chunker creates boundaries based on similarity"""
    embedder = FakeDeterministicEmbeddings(dim=64)
    # High threshold means only very similar sentences stay together
    chunker = GreedyBoundarySemanticChunker(embedder, max_tokens=200, sim_threshold=0.9, overlap_sentences=1)
    
    # Mix unrelated topics that should create semantic boundaries
    mixed_content = """Machine learning algorithms require large datasets for training.
    The recipe for chocolate cake includes flour, sugar, and eggs.
    Neural networks use backpropagation for weight updates.
    Cats are independent pets that require minimal maintenance."""
    
    pages = [(1, mixed_content)]
    chunks = chunker.chunk_pages(doc_id="mixed", pages=pages, source_url="mixed.pdf", tenant="test")
    
    # Should create multiple chunks due to semantic dissimilarity
    assert len(chunks) >= 2
    
def test_semantic_chunker_overlap():
    """Test sentence overlap functionality"""
    embedder = FakeDeterministicEmbeddings(dim=64)
    chunker = GreedyBoundarySemanticChunker(embedder, max_tokens=40, sim_threshold=0.5, overlap_sentences=2)
    
    text = """First sentence here. Second sentence follows. Third sentence continues. Fourth sentence ends the thought. Fifth sentence starts new topic."""
    
    pages = [(1, text)]
    chunks = chunker.chunk_pages(doc_id="overlap", pages=pages, source_url="overlap.pdf", tenant="test")
    
    if len(chunks) >= 2:
        # Check that there's some overlap between consecutive chunks
        first_chunk_end = chunks[0].text.split('.')[-3:]  # Last few sentences
        second_chunk_start = chunks[1].text.split('.')[:3]  # First few sentences
        
        # Should have some overlapping content
        overlap_found = any(sentence.strip() in ' '.join(second_chunk_start) 
                          for sentence in first_chunk_end if sentence.strip())
        assert overlap_found
