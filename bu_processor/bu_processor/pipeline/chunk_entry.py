from __future__ import annotations
from typing import List, Tuple
import uuid

from bu_processor.config import settings
from bu_processor.factories import make_embedder
from bu_processor.semantic.greedy_boundary_chunker import GreedyBoundarySemanticChunker
from bu_processor.models.chunk import DocumentChunk

def chunk_document_pages(*,
                         pages: List[Tuple[int, str]],  # [(page_no, page_text)]
                         source_url: str | None,
                         tenant: str | None,
                         prefer_semantic: bool = True,
                         doc_id: str | None = None) -> List[DocumentChunk]:
    """
    One entry point. Always produce DocumentChunk[] with metadata set.
    
    This function provides a unified interface for document chunking that can
    use either semantic chunking (when available and preferred) or fall back
    to simple chunking.
    
    Args:
        pages: List of (page_number, page_text) tuples
        source_url: Optional source URL for metadata
        tenant: Optional tenant identifier for metadata
        prefer_semantic: Whether to prefer semantic chunking when available
        doc_id: Optional document ID, will be generated if not provided
        
    Returns:
        List of DocumentChunk objects with rich metadata
    """
    _doc_id = doc_id or str(uuid.uuid4())

    if prefer_semantic and settings.ENABLE_SEMANTIC_CHUNKING:
        try:
            embedder = make_embedder()
            chunker = GreedyBoundarySemanticChunker(
                embedder=embedder,
                max_tokens=settings.SEMANTIC_MAX_TOKENS,
                sim_threshold=settings.SEMANTIC_SIM_THRESHOLD,
                overlap_sentences=settings.SEMANTIC_OVERLAP_SENTENCES,
            )
            return chunker.chunk_pages(
                doc_id=_doc_id, 
                pages=pages, 
                source_url=source_url, 
                tenant=tenant
            )
        except Exception as e:
            # Log the error and fall back to simple chunking
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Semantic chunking failed, falling back to simple chunking: {e}")

    # Fallback: simple chunking
    from bu_processor.chunking.simple import simple_fixed_chunks_from_pages
    return simple_fixed_chunks_from_pages(
        doc_id=_doc_id,
        pages=pages,
        source_url=source_url,
        tenant=tenant,
        max_tokens=settings.SEMANTIC_MAX_TOKENS,  # Use same token budget
        overlap_sentences=settings.SEMANTIC_OVERLAP_SENTENCES
    )

def chunk_document_text(*,
                        text: str,
                        source_url: str | None = None,
                        tenant: str | None = None,
                        prefer_semantic: bool = True,
                        doc_id: str | None = None) -> List[DocumentChunk]:
    """
    Convenience method to chunk a single text string.
    
    Args:
        text: Text content to chunk
        source_url: Optional source URL for metadata
        tenant: Optional tenant identifier for metadata
        prefer_semantic: Whether to prefer semantic chunking when available
        doc_id: Optional document ID, will be generated if not provided
        
    Returns:
        List of DocumentChunk objects
    """
    pages = [(1, text)]
    return chunk_document_pages(
        pages=pages,
        source_url=source_url,
        tenant=tenant,
        prefer_semantic=prefer_semantic,
        doc_id=doc_id
    )

def get_chunking_stats(chunks: List[DocumentChunk]) -> dict:
    """
    Get statistics about chunking results.
    
    Args:
        chunks: List of chunks to analyze
        
    Returns:
        Dictionary with chunking statistics
    """
    if not chunks:
        return {"total_chunks": 0}
    
    import numpy as np
    
    token_counts = [chunk.meta.get("token_count", 0) for chunk in chunks]
    sentence_counts = [chunk.meta.get("sentence_count", 0) for chunk in chunks]
    chunking_methods = [chunk.meta.get("chunking_method", "unknown") for chunk in chunks]
    
    stats = {
        "total_chunks": len(chunks),
        "chunking_methods": list(set(chunking_methods)),
        "avg_tokens_per_chunk": float(np.mean(token_counts)) if token_counts else 0,
        "max_tokens_per_chunk": max(token_counts) if token_counts else 0,
        "min_tokens_per_chunk": min(token_counts) if token_counts else 0,
        "avg_sentences_per_chunk": float(np.mean(sentence_counts)) if sentence_counts else 0,
        "pages_covered": len(set(chunk.page_start for chunk in chunks if chunk.page_start)),
        "sections_covered": len(set(chunk.section for chunk in chunks if chunk.section)),
        "chunk_types": list(set(chunk.chunk_type for chunk in chunks)),
        "avg_importance": float(np.mean([chunk.importance_score for chunk in chunks])),
        "total_text_length": sum(len(chunk.text) for chunk in chunks)
    }
    
    return stats
