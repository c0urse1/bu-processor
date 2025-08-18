from __future__ import annotations
from typing import List, Optional, Dict, Any
import numpy as np
import uuid

from .embeddings import EmbeddingsBackend
from .sentences import sentence_split_with_offsets
from .structure import detect_headings, assign_section_for_offset
from .tokens import approx_token_count  # your existing helper
from ..models.chunk import DocumentChunk

def _norm(v: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, 1e-8, None)

class GreedyBoundarySemanticChunker:
    """
    Semantic chunker that combines:
    - Embedding similarity to continue a chunk
    - Hard boundaries at page breaks & heading changes  
    - Token budget per chunk; small overlap between chunks
    - Rich metadata population during chunking
    
    Key properties:
    - Semantic: grows a chunk while cosine similarity ≥ threshold
    - Structure-aware: hard break on page change or heading change
    - Budget-aware: enforces max_tokens, with small overlap for continuity
    - Metadata is set now, not later
    """
    
    def __init__(self,
                 embedder: EmbeddingsBackend,
                 max_tokens: int = 480,
                 sim_threshold: float = 0.62,
                 overlap_sentences: int = 1):
        """
        Initialize the semantic chunker.
        
        Args:
            embedder: Embeddings backend for computing sentence similarities
            max_tokens: Maximum tokens per chunk (hard limit)
            sim_threshold: Minimum cosine similarity to continue chunk (0.0-1.0)
            overlap_sentences: Number of sentences to overlap between chunks
        """
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.sim_threshold = sim_threshold
        self.overlap = overlap_sentences

    def chunk_pages(self, *,
                    doc_id: str,
                    pages: List[tuple],              # [(page_no, page_text), ...]
                    source_url: Optional[str] = None,
                    tenant: Optional[str] = None) -> List[DocumentChunk]:
        """
        Chunk document pages using semantic similarity and structure awareness.
        
        Args:
            doc_id: Document identifier
            pages: List of (page_number, page_text) tuples
            source_url: Optional source URL for metadata
            tenant: Optional tenant identifier for metadata
            
        Returns:
            List of DocumentChunk objects with rich metadata
        """
        # 1) Detect headings per page
        page_heads = {pno: detect_headings(txt) for pno, txt in pages}
        
        # 2) Get sentence list with page/offset information
        sents = sentence_split_with_offsets(pages)  # [(sent, page_no, offset)]
        if not sents:
            return []

        # 3) Generate embeddings for all sentences
        texts = [s[0] for s in sents]
        embs = self.embedder.encode(texts)
        embs = _norm(embs)

        chunks: List[DocumentChunk] = []

        def heading_for(i: int) -> tuple[str, str]:
            """Get section and title for sentence at index i."""
            _, pno, off = sents[i]
            sec, title = assign_section_for_offset(off, page_heads.get(pno, []))
            return sec, title

        def hard_boundary(i: int, j: int) -> bool:
            """Check if there should be a hard boundary between sentences i and j."""
            # Boundary if page changes or heading changes between i (prev) and j (next)
            _, p_prev, _ = sents[i]
            _, p_next, _ = sents[j]
            if p_prev != p_next:
                return True
            sec_i, title_i = heading_for(i)
            sec_j, title_j = heading_for(j)
            return (sec_i, title_i) != (sec_j, title_j)

        def build_heading_path(i: int) -> List[str]:
            """Build hierarchical heading path for sentence at index i."""
            sec, title = heading_for(i)
            path = []
            if sec:
                # For numbered sections like "2.1", create hierarchy
                parts = sec.split('.')
                if len(parts) > 1:
                    # Build path like ["2", "2.1"] 
                    for j in range(len(parts)):
                        path.append('.'.join(parts[:j+1]))
                else:
                    path.append(sec)
            if title:
                path.append(title)
            return path

        def create_chunk(start_idx: int, end_idx: int) -> DocumentChunk:
            """Create a DocumentChunk from sentence range [start_idx, end_idx)."""
            chunk_sents = [x[0] for x in sents[start_idx:end_idx]]
            text = " ".join(chunk_sents).strip()
            
            # Page range
            p_start = sents[start_idx][1]
            p_end = sents[end_idx-1][1] if end_idx > start_idx else p_start
            
            # Section information
            sec, title = heading_for(start_idx)
            section_str = f"{sec} {title}".strip() if title else (sec or None)
            heading_path = build_heading_path(start_idx)
            
            # Character span information (within first page)
            start_char = sents[start_idx][2]  # offset in page
            end_char = sents[end_idx-1][2] + len(sents[end_idx-1][0]) if end_idx > start_idx else start_char
            
            # Calculate importance score based on section depth and content
            importance_score = 1.0
            if sec:
                # Higher importance for higher-level sections
                depth = len(sec.split('.'))
                importance_score = max(0.5, 1.0 - (depth - 1) * 0.1)
            
            # Detect if this is likely an important section (introduction, conclusion, etc.)
            if title:
                important_keywords = ['introduction', 'conclusion', 'summary', 'executive', 'overview']
                if any(keyword in title.lower() for keyword in important_keywords):
                    importance_score += 0.2
            
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                text=text,
                doc_id=doc_id,
                page_start=p_start,
                page_end=p_end,
                section=section_str,
                heading_path=heading_path,
                char_span=(start_char, end_char) if p_start == p_end else None,
                chunk_type="semantic",
                importance_score=min(1.5, importance_score),  # Cap at 1.5
                heading_text=title,
                meta={
                    "source_url": source_url,
                    "tenant": tenant,
                    "sentence_count": end_idx - start_idx,
                    "token_count": sum(approx_token_count(sents[j][0]) for j in range(start_idx, end_idx)),
                    "avg_similarity": None,  # Will be set if we track it
                    "chunking_method": "greedy_boundary_semantic"
                }
            )
            return chunk

        # Main chunking loop
        start = 0
        budget = approx_token_count(sents[0][0])
        cur_centroid = embs[0].copy()
        similarities = []

        for i in range(1, len(sents)):
            # Decide if we add sents[i] to current chunk
            sim = float(np.dot(embs[i], cur_centroid))
            similarities.append(sim)
            cand_budget = budget + approx_token_count(sents[i][0])
            
            # Check for chunk boundary conditions
            should_break = (
                hard_boundary(i-1, i) or           # Structure change
                sim < self.sim_threshold or        # Semantic dissimilarity
                cand_budget > self.max_tokens      # Token budget exceeded
            )
            
            if should_break:
                # Finalize chunk [start, i)
                chunk = create_chunk(start, i)
                if similarities:
                    chunk.meta["avg_similarity"] = float(np.mean(similarities))
                chunks.append(chunk)

                # Start new window with overlap
                overlap_start = max(i - self.overlap, 0)
                start = overlap_start
                
                # Recalculate budget and centroid for new chunk
                budget = sum(approx_token_count(sents[j][0]) for j in range(start, i+1))
                if start < i:
                    cur_centroid = _norm(np.mean(embs[start:i+1], axis=0, keepdims=True))[0]
                else:
                    cur_centroid = embs[i].copy()
                similarities = []
            else:
                # Continue current chunk
                budget = cand_budget
                # Update centroid (incremental average)
                chunk_size = i - start + 1
                cur_centroid = (cur_centroid * (chunk_size - 1) + embs[i]) / chunk_size
                cur_centroid = _norm(cur_centroid.reshape(1, -1))[0]

        # Handle final chunk (tail)
        if start < len(sents):
            chunk = create_chunk(start, len(sents))
            if similarities:
                chunk.meta["avg_similarity"] = float(np.mean(similarities))
            chunks.append(chunk)

        return chunks

    def chunk_text(self, 
                   text: str, 
                   doc_id: str,
                   source_url: Optional[str] = None,
                   tenant: Optional[str] = None) -> List[DocumentChunk]:
        """
        Convenience method to chunk a single text string.
        
        Args:
            text: Text content to chunk
            doc_id: Document identifier
            source_url: Optional source URL
            tenant: Optional tenant identifier
            
        Returns:
            List of DocumentChunk objects
        """
        pages = [(1, text)]
        return self.chunk_pages(
            doc_id=doc_id,
            pages=pages,
            source_url=source_url,
            tenant=tenant
        )

    def get_chunk_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Get statistics about the chunking results.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {}
        
        token_counts = [chunk.meta.get("token_count", 0) for chunk in chunks]
        similarities = [chunk.meta.get("avg_similarity") for chunk in chunks if chunk.meta.get("avg_similarity") is not None]
        
        stats = {
            "total_chunks": len(chunks),
            "avg_tokens_per_chunk": float(np.mean(token_counts)) if token_counts else 0,
            "max_tokens_per_chunk": max(token_counts) if token_counts else 0,
            "min_tokens_per_chunk": min(token_counts) if token_counts else 0,
            "avg_similarity": float(np.mean(similarities)) if similarities else None,
            "pages_covered": len(set(chunk.page_start for chunk in chunks if chunk.page_start)),
            "sections_covered": len(set(chunk.section for chunk in chunks if chunk.section)),
            "chunk_types": list(set(chunk.chunk_type for chunk in chunks)),
            "avg_importance": float(np.mean([chunk.importance_score for chunk in chunks]))
        }
        
        return stats
