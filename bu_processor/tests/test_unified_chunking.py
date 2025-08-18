"""
Tests for unified document chunking entry point.

These tests validate the single entry point chunking system that decides
between semantic and simple chunking based on configuration and availability.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from bu_processor.chunking import chunk_document, simple_fixed_chunks, sentence_split


def test_sentence_split_basic():
    """Test basic sentence splitting functionality."""
    text = "This is sentence one. This is sentence two! What about sentence three?"
    sentences = sentence_split(text)
    
    assert len(sentences) == 3
    assert "This is sentence one" in sentences[0]
    assert "This is sentence two" in sentences[1]
    assert "What about sentence three" in sentences[2]


def test_simple_fixed_chunks():
    """Test simple chunking fallback."""
    sentences = [
        "Short sentence.",
        "Another short sentence.",
        "This is a much longer sentence that should probably be in its own chunk.",
        "Final sentence."
    ]
    
    chunks = simple_fixed_chunks(sentences, max_tokens=20, overlap_sentences=1)
    
    assert len(chunks) >= 2  # Should split due to token budget
    assert all(chunk.strip() for chunk in chunks)  # All chunks should be non-empty


def test_chunk_document_empty_input():
    """Test chunking with empty input."""
    assert chunk_document("") == []
    assert chunk_document("   ") == []
    assert chunk_document(None) == []


def test_chunk_document_semantic_disabled():
    """Test chunking when semantic is explicitly disabled."""
    text = "This is a test. Another sentence here. Final sentence."
    
    chunks = chunk_document(text, enable_semantic=False, max_tokens=50)
    
    assert len(chunks) >= 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_document_semantic_enabled_fallback():
    """Test that semantic chunking falls back gracefully when dependencies fail."""
    text = "This is about cats. Cats are great pets. Finance is important for business."
    
    # Mock the SentenceTransformer import to fail
    with patch('bu_processor.chunking.SbertEmbeddings', side_effect=ImportError("No sentence-transformers")):
        chunks = chunk_document(text, enable_semantic=True, max_tokens=50)
    
    # Should fall back to simple chunking
    assert len(chunks) >= 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_document_semantic_success():
    """Test semantic chunking when dependencies are available."""
    text = "Cats are amazing pets. They love to sleep. Finance is important for businesses. Banking helps companies."
    
    # Mock the semantic components to work
    from bu_processor.semantic.testing import FakeDeterministicEmbeddings
    
    with patch('bu_processor.chunking.SbertEmbeddings') as mock_sbert:
        # Make SbertEmbeddings return our fake embedder
        mock_embedder = FakeDeterministicEmbeddings(dim=384)
        mock_sbert.return_value = mock_embedder
        
        chunks = chunk_document(text, enable_semantic=True, max_tokens=50, sim_threshold=0.6)
    
    assert len(chunks) >= 2  # Should create multiple chunks for different topics
    assert all(chunk.strip() for chunk in chunks)
    
    # Verify semantic chunking was attempted
    mock_sbert.assert_called_once()


def test_chunk_document_config_integration():
    """Test that chunking respects configuration when available."""
    text = "Test sentence one. Test sentence two. Test sentence three."
    
    # Mock config to disable semantic chunking
    mock_config = MagicMock()
    mock_config.semantic.enable_semantic_chunking = False
    
    with patch('bu_processor.chunking.get_config', return_value=mock_config):
        chunks = chunk_document(text, enable_semantic=None)  # Should use config
    
    # Should use simple chunking due to config
    assert len(chunks) >= 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_document_validation_fallback():
    """Test that invalid semantic results trigger fallback."""
    text = "Test sentence for validation."
    
    # Mock semantic chunking to return empty result
    with patch('bu_processor.chunking.semantic_segment_sentences', return_value=[]):
        chunks = chunk_document(text, enable_semantic=True)
    
    # Should fall back to simple chunking
    assert len(chunks) >= 1
    assert chunks[0].strip() == text


def test_chunk_document_token_parameters():
    """Test that token parameters are respected."""
    text = "Word " * 100  # 100 words, roughly 77 tokens
    
    chunks = chunk_document(text, enable_semantic=False, max_tokens=30)
    
    # Should create multiple chunks due to token limit
    assert len(chunks) >= 2
    
    # Each chunk should respect approximate token limit
    from bu_processor.semantic.tokens import approx_token_count
    for chunk in chunks:
        token_count = approx_token_count(chunk)
        # Allow some flexibility since we chunk by sentences
        assert token_count <= 50  # Reasonable upper bound


def test_chunk_document_overlap():
    """Test sentence overlap between chunks."""
    sentences = ["First sentence.", "Second sentence.", "Third sentence.", "Fourth sentence."]
    text = " ".join(sentences)
    
    chunks = chunk_document(text, enable_semantic=False, max_tokens=10, overlap_sentences=1)
    
    if len(chunks) >= 2:
        # Check that there's some overlap (exact overlap depends on sentence splitting)
        first_words = set(chunks[0].split())
        second_words = set(chunks[1].split())
        overlap = len(first_words.intersection(second_words))
        assert overlap > 0  # Should have some overlapping words


@pytest.mark.parametrize("enable_semantic,expected_method", [
    (True, "semantic_attempted"),
    (False, "simple_only"),
    (None, "config_dependent")
])
def test_chunk_document_method_selection(enable_semantic, expected_method):
    """Test that the correct chunking method is selected based on parameters."""
    text = "Test sentence for method selection."
    
    if expected_method == "semantic_attempted":
        # Should attempt semantic (may fall back)
        with patch('bu_processor.chunking.SbertEmbeddings', side_effect=ImportError()) as mock_sbert:
            chunks = chunk_document(text, enable_semantic=enable_semantic)
            # ImportError should trigger fallback, but SbertEmbeddings should be called
            mock_sbert.assert_called_once()
    
    elif expected_method == "simple_only":
        # Should skip semantic entirely
        with patch('bu_processor.chunking.SbertEmbeddings') as mock_sbert:
            chunks = chunk_document(text, enable_semantic=enable_semantic)
            # SbertEmbeddings should never be called
            mock_sbert.assert_not_called()
    
    elif expected_method == "config_dependent":
        # Should check config
        with patch('bu_processor.chunking.get_config') as mock_get_config:
            mock_config = MagicMock()
            mock_config.semantic.enable_semantic_chunking = True
            mock_get_config.return_value = mock_config
            
            with patch('bu_processor.chunking.SbertEmbeddings', side_effect=ImportError()):
                chunks = chunk_document(text, enable_semantic=enable_semantic)
            
            mock_get_config.assert_called_once()
    
    # All methods should produce valid output
    assert len(chunks) >= 1
    assert all(chunk.strip() for chunk in chunks)


def test_unified_chunking_no_duplication():
    """Test that unified chunking doesn't cause duplication issues."""
    text = "Sentence one about cats. Sentence two about cats. Sentence three about finance."
    
    # Test multiple calls produce consistent results
    chunks1 = chunk_document(text, enable_semantic=False, max_tokens=50)
    chunks2 = chunk_document(text, enable_semantic=False, max_tokens=50)
    
    assert len(chunks1) == len(chunks2)
    assert chunks1 == chunks2  # Should be identical


def test_logging_behavior():
    """Test that logging works correctly for different scenarios."""
    import logging
    from bu_processor.chunking import logger
    
    # Capture log messages
    with patch.object(logger, 'debug') as mock_debug, \
         patch.object(logger, 'info') as mock_info, \
         patch.object(logger, 'warning') as mock_warning:
        
        text = "Test logging behavior."
        
        # Test semantic disabled logging
        chunk_document(text, enable_semantic=False)
        mock_debug.assert_any_call("Semantic chunking disabled, using simple chunking")
        
        # Test semantic failure logging
        with patch('bu_processor.chunking.SbertEmbeddings', side_effect=ImportError("Test error")):
            chunk_document(text, enable_semantic=True)
        
        # Should log the fallback warning
        mock_warning.assert_any_call("Semantic chunking dependencies not available, falling back to simple", error="Test error")


# Integration test for real functionality (can be skipped in CI)
@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("RUN_REAL_SEMANTIC"), reason="set RUN_REAL_SEMANTIC=1 to run")
def test_chunk_document_real_semantic():
    """Integration test with real semantic chunking (requires sentence-transformers)."""
    text = "Cats are wonderful pets that bring joy. Dogs are also great companions. Financial planning is crucial for business success. Investment strategies require careful analysis."
    
    chunks = chunk_document(text, enable_semantic=True, max_tokens=50, sim_threshold=0.6)
    
    # Should produce multiple topic-based chunks
    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)
    
    # Should separate pet topics from finance topics
    combined_text = " ".join(chunks).lower()
    assert "cat" in combined_text or "dog" in combined_text
    assert "financial" in combined_text or "investment" in combined_text
