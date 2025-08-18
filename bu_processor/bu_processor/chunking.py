"""
Unified document chunking entry point.

This module provides a single entry point for all document chunking operations,
deciding between semantic and simple chunking based on configuration and fallbacks.
"""

import logging
from typing import List, Optional
from .semantic.embeddings import SbertEmbeddings
from .semantic.chunker import semantic_segment_sentences
from .semantic.tokens import approx_token_count

# Dedicated logger for semantic operations
logger = logging.getLogger("bu_processor.semantic")

def sentence_split(text: str) -> List[str]:
    """
    Split text into sentences using simple heuristics.
    
    This is a fallback implementation that can be replaced with 
    more sophisticated sentence splitting if needed.
    """
    import re
    
    # Simple sentence splitting - can be enhanced later
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def simple_fixed_chunks(sentences: List[str], max_tokens: int = 480, overlap_sentences: int = 1) -> List[str]:
    """
    Simple chunking fallback that groups sentences by token budget.
    
    Args:
        sentences: List of sentences to chunk
        max_tokens: Maximum tokens per chunk
        overlap_sentences: Number of sentences to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = approx_token_count(sentence)
        
        # If adding this sentence would exceed the limit, finalize current chunk
        if current_chunk and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap
            if overlap_sentences > 0 and len(current_chunk) >= overlap_sentences:
                current_chunk = current_chunk[-overlap_sentences:]
                current_tokens = sum(approx_token_count(s) for s in current_chunk)
            else:
                current_chunk = []
                current_tokens = 0
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def chunk_document(text: str, enable_semantic: Optional[bool] = None, 
                  max_tokens: int = 480, sim_threshold: float = 0.62,
                  overlap_sentences: int = 1, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[str]:
    """
    Single entry point for document chunking.
    
    Decides between semantic and simple chunking based on configuration and availability.
    Always falls back gracefully to simple chunking if semantic fails.
    
    Args:
        text: Input text to chunk
        enable_semantic: Override for semantic chunking (None = use config)
        max_tokens: Maximum tokens per chunk
        sim_threshold: Semantic similarity threshold for chunking
        overlap_sentences: Number of sentences to overlap between chunks
        model_name: Name of the sentence transformer model to use
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Split into sentences first
    sentences = sentence_split(text)
    if not sentences:
        return []
    
    # Determine if we should use semantic chunking
    use_semantic = enable_semantic
    if use_semantic is None:
        try:
            # Try to get from config, fall back to True
            from .core.config import get_config
            config = get_config()
            use_semantic = getattr(config.semantic, 'enable_semantic_chunking', True)
        except Exception:
            # If config is not available, default to True and let semantic chunking fail gracefully
            use_semantic = True
            logger.debug("Could not load semantic config, defaulting to enabled")
    
    # If semantic chunking is disabled, use simple chunking
    if not use_semantic:
        logger.debug("Semantic chunking disabled, using simple chunking")
        return simple_fixed_chunks(sentences, max_tokens=max_tokens, overlap_sentences=overlap_sentences)
    
    # Try semantic chunking with graceful fallback
    try:
        logger.debug(f"Attempting semantic chunking - {len(sentences)} sentences")
        embedder = SbertEmbeddings(model_name=model_name)
        chunks = semantic_segment_sentences(
            sentences,
            embedder=embedder,
            max_tokens=max_tokens,
            sim_threshold=sim_threshold,
            overlap_sentences=overlap_sentences,
        )
        
        # Validation: ensure semantic chunking produced reasonable results
        if not chunks:
            logger.warning("Semantic chunking produced no chunks, falling back to simple")
            return simple_fixed_chunks(sentences, max_tokens=max_tokens, overlap_sentences=overlap_sentences)
        
        total_semantic_tokens = sum(approx_token_count(c) for c in chunks)
        if total_semantic_tokens < 5:
            logger.warning(f"Semantic chunking produced very small output, falling back to simple - total_tokens={total_semantic_tokens}")
            return simple_fixed_chunks(sentences, max_tokens=max_tokens, overlap_sentences=overlap_sentences)
        
        logger.info(f"Semantic chunking successful - {len(chunks)} chunks, {total_semantic_tokens} total tokens")
        return chunks
        
    except ImportError as e:
        logger.warning(f"Semantic chunking dependencies not available, falling back to simple - error: {str(e)}")
        return simple_fixed_chunks(sentences, max_tokens=max_tokens, overlap_sentences=overlap_sentences)
    
    except Exception as e:
        logger.exception("Semantic chunking failed, falling back to simple")
        return simple_fixed_chunks(sentences, max_tokens=max_tokens, overlap_sentences=overlap_sentences)
