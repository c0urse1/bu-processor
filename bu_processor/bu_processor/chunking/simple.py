"""Simple chunking module for fallback when semantic chunking is not available."""

from __future__ import annotations
from typing import List, Tuple
import uuid

from ..models.chunk import DocumentChunk
from ..semantic.sentences import sentence_split_with_offsets
from ..semantic.tokens import approx_token_count

def simple_fixed_chunks_from_pages(
    *,
    doc_id: str,
    pages: List[Tuple[int, str]],
    source_url: str | None = None,
    tenant: str | None = None,
    max_tokens: int = 480,
    overlap_sentences: int = 1
) -> List[DocumentChunk]:
    """
    Simple chunking that groups sentences by token budget without semantic analysis.
    
    Args:
        doc_id: Document identifier
        pages: List of (page_number, page_text) tuples
        source_url: Optional source URL for metadata
        tenant: Optional tenant identifier
        max_tokens: Maximum tokens per chunk
        overlap_sentences: Number of sentences to overlap between chunks
        
    Returns:
        List of DocumentChunk objects
    """
    if not pages:
        return []
    
    # Get sentences with page/offset information
    sentences_with_offsets = sentence_split_with_offsets(pages)
    if not sentences_with_offsets:
        return []
    
    chunks = []
    current_sentences = []
    current_tokens = 0
    chunk_counter = 0
    
    for sentence, page_no, offset in sentences_with_offsets:
        sentence_tokens = approx_token_count(sentence)
        
        # If adding this sentence would exceed the limit, finalize current chunk
        if current_sentences and current_tokens + sentence_tokens > max_tokens:
            # Create chunk from current sentences
            chunk = _create_simple_chunk(
                chunk_id=f"{doc_id}_simple_{chunk_counter}",
                sentences_info=current_sentences,
                doc_id=doc_id,
                source_url=source_url,
                tenant=tenant
            )
            chunks.append(chunk)
            chunk_counter += 1
            
            # Start new chunk with overlap
            if overlap_sentences > 0 and len(current_sentences) >= overlap_sentences:
                current_sentences = current_sentences[-overlap_sentences:]
                current_tokens = sum(approx_token_count(s[0]) for s in current_sentences)
            else:
                current_sentences = []
                current_tokens = 0
        
        current_sentences.append((sentence, page_no, offset))
        current_tokens += sentence_tokens
    
    # Add final chunk
    if current_sentences:
        chunk = _create_simple_chunk(
            chunk_id=f"{doc_id}_simple_{chunk_counter}",
            sentences_info=current_sentences,
            doc_id=doc_id,
            source_url=source_url,
            tenant=tenant
        )
        chunks.append(chunk)
    
    return chunks

def _create_simple_chunk(
    chunk_id: str,
    sentences_info: List[Tuple[str, int, int]],  # [(sentence, page_no, offset), ...]
    doc_id: str,
    source_url: str | None = None,
    tenant: str | None = None
) -> DocumentChunk:
    """Create a DocumentChunk from sentence information."""
    
    if not sentences_info:
        return DocumentChunk(
            chunk_id=chunk_id,
            text="",
            doc_id=doc_id,
            chunk_type="simple",
            meta={"source_url": source_url, "tenant": tenant}
        )
    
    # Extract information
    text = " ".join(s[0] for s in sentences_info)
    pages = [s[1] for s in sentences_info]
    page_start = min(pages)
    page_end = max(pages)
    
    # Calculate token and sentence counts
    token_count = sum(approx_token_count(s[0]) for s in sentences_info)
    sentence_count = len(sentences_info)
    
    # Character span for single-page chunks
    char_span = None
    if page_start == page_end and sentences_info:
        start_char = sentences_info[0][2]  # First sentence offset
        last_sentence = sentences_info[-1]
        end_char = last_sentence[2] + len(last_sentence[0])
        char_span = (start_char, end_char)
    
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        doc_id=doc_id,
        page_start=page_start,
        page_end=page_end,
        char_span=char_span,
        chunk_type="simple",
        importance_score=1.0,  # Default importance for simple chunks
        meta={
            "source_url": source_url,
            "tenant": tenant,
            "token_count": token_count,
            "sentence_count": sentence_count,
            "chunking_method": "simple_fixed_chunks"
        }
    )
    
    return chunk
