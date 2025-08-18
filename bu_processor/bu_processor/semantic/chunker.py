# bu_processor/semantic/chunker.py
from __future__ import annotations
from typing import List, Optional
import numpy as np
from .embeddings import EmbeddingsBackend
from .tokens import approx_token_count

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized vectors."""
    # a and b should be L2-normalized vectors
    return float(np.dot(a, b))

def _mean(vecs: List[np.ndarray]) -> np.ndarray:
    """Compute normalized mean of a list of vectors."""
    m = np.mean(np.stack(vecs, axis=0), axis=0)
    n = np.linalg.norm(m)
    return m / max(n, 1e-8)

def semantic_segment_sentences(
    sentences: List[str],
    embedder: EmbeddingsBackend,
    max_tokens: int = 480,
    sim_threshold: float = 0.62,
    overlap_sentences: int = 1,
    min_chunk_sentences: int = 1,
) -> List[str]:
    """
    Greedy segmentation by semantic cohesion:
    - compute embeddings per sentence (normalized)
    - grow a chunk while cosine(current_sentence, running_centroid) >= threshold
      and token budget not exceeded
    - when threshold/budget fails, cut and start a new chunk
    - add sentence-overlap between chunks for context
    
    Args:
        sentences: List of sentences to segment
        embedder: Embedding backend to use
        max_tokens: Maximum tokens per chunk
        sim_threshold: Cosine similarity threshold to running centroid
        overlap_sentences: Number of sentences to overlap between chunks
        min_chunk_sentences: Minimum sentences per chunk
        
    Returns:
        List of semantic chunks (as strings)
    """
    if not sentences:
        return []

    embs = embedder.encode(sentences)
    # Normalize in case backend didn't
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-8, None)

    chunks: List[str] = []
    cur_idx_start = 0
    cur_vecs = [embs[0]]
    cur_tokens = approx_token_count(sentences[0])

    for i in range(1, len(sentences)):
        centroid = _mean(cur_vecs)
        sim = _cos_sim(embs[i], centroid)
        cand_tokens = cur_tokens + approx_token_count(sentences[i])

        if sim >= sim_threshold and cand_tokens <= max_tokens:
            # keep growing
            cur_vecs.append(embs[i])
            cur_tokens = cand_tokens
        else:
            # finalize chunk
            end = i  # exclusive
            # enforce minimum sentences per chunk
            if end - cur_idx_start < min_chunk_sentences and i < len(sentences) - 1:
                # force-include one more sentence
                cur_vecs.append(embs[i])
                cur_tokens += approx_token_count(sentences[i])
                i += 1  # will be incremented by loop
                end = i

            text = " ".join(sentences[cur_idx_start:end]).strip()
            if text:
                chunks.append(text)

            # start new chunk with overlap
            new_start = max(end - overlap_sentences, 0)
            cur_idx_start = new_start
            cur_vecs = [embs[j] for j in range(new_start, i + 1)]
            cur_tokens = approx_token_count(" ".join(sentences[new_start:i+1]))

    # last chunk
    last_text = " ".join(sentences[cur_idx_start:]).strip()
    if last_text:
        chunks.append(last_text)

    return chunks
