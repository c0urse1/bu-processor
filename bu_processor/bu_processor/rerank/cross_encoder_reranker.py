from __future__ import annotations
from typing import List, Optional
import numpy as np

from bu_processor.retrieval.models import RetrievalHit

class CrossEncoderReranker:
    """
    HuggingFace Sentence-Transformers CrossEncoder reranker.
    Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, high quality).
    Scores pairs (query, passage) and sorts by descending score.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: Optional[str] = None, batch_size: int = 32):
        # Lazy import to avoid cost outside production paths
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def rerank(self, query: str, hits: List[RetrievalHit], top_k: Optional[int] = None) -> List[RetrievalHit]:
        if not hits:
            return hits

        pairs = [(query, h.text) for h in hits]
        # CrossEncoder returns higher = more relevant (regression score)
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        scores = np.asarray(scores, dtype=np.float32)
        # Combine with incoming score only if you want a blend; by default we trust CrossEncoder
        order = np.argsort(-scores)  # descending
        ranked = [hits[i] for i in order]

        # Write the CE score back (optional but useful for tracing)
        for i, idx in enumerate(order):
            ranked[i].metadata = dict(ranked[i].metadata)  # avoid accidental shared dict
            ranked[i].metadata["ce_score"] = float(scores[idx])

        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked
