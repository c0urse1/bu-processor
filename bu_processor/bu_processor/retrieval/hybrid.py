# bu_processor/retrieval/hybrid.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np

from bu_processor.ports import EmbeddingsBackend
from bu_processor.retrieval.models import RetrievalHit
from bu_processor.retrieval.dense import DenseKnnRetriever
from bu_processor.retrieval.bm25 import Bm25Index
from bu_processor.retrieval.fusion import rrf_fuse, weighted_sum_fuse
from bu_processor.retrieval.mmr import mmr_select

if TYPE_CHECKING:
    from bu_processor.ports import Reranker, QueryAwareSummarizer

class HybridRetriever:
    """
    Combines Dense + BM25 using RRF (default) or weighted sum.
    Optional MMR diversification on final list.
    Optional reranking and query-aware summarization.
    """
    def __init__(self,
                 dense: DenseKnnRetriever,
                 bm25: Bm25Index,
                 embedder: EmbeddingsBackend,
                 fusion: str = "rrf",           # "rrf" or "weighted"
                 alpha_dense: float = 0.5,
                 alpha_bm25: float = 0.5,
                 use_mmr: bool = True,
                 mmr_lambda: float = 0.65,
                 reranker: Optional["Reranker"] = None,
                 summarizer: Optional["QueryAwareSummarizer"] = None,
                 summary_tokens: int = 160):
        self.dense = dense
        self.bm25 = bm25
        self.embedder = embedder
        self.fusion = fusion
        self.alpha_dense = alpha_dense
        self.alpha_bm25 = alpha_bm25
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.reranker = reranker
        self.summarizer = summarizer
        self.summary_tokens = summary_tokens

    def retrieve(self, query: str, final_top_k: int = 5,
                 top_k_dense: int = 8, top_k_bm25: int = 12,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[RetrievalHit]:
        d_hits = self.dense.retrieve(query, top_k=top_k_dense, metadata_filter=metadata_filter)
        b_hits = self.bm25.query(query, top_k=top_k_bm25, metadata_filter=metadata_filter)

        if self.fusion == "weighted":
            fused = weighted_sum_fuse(d_hits, b_hits, self.alpha_dense, self.alpha_bm25)
        else:
            fused = rrf_fuse([d_hits, b_hits])

        # truncate before MMR to keep compute tight
        fused = fused[: max(final_top_k * 4, final_top_k + 5)]

        candidates = fused
        if self.use_mmr:
            # MMR requires vectors; compute from embedder on demand (no vector store needed)
            qv = self.embedder.encode([query])[0]
            cache: Dict[str, np.ndarray] = {}

            def get_vec(h: RetrievalHit) -> np.ndarray:
                if h.id not in cache:
                    cache[h.id] = self.embedder.encode([h.text])[0]
                return cache[h.id]

            candidates = mmr_select(qv, fused, get_vec=get_vec, top_k=final_top_k * 2, lambda_mult=self.mmr_lambda)
        
        # (1) Cross-Encoder (or heuristic) rerank over the candidates
        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates, top_k=final_top_k)
        else:
            candidates = candidates[:final_top_k]

        # (2) Optional query-aware summarization (store in metadata)
        if self.summarizer is not None:
            for h in candidates:
                summary = self.summarizer.summarize(query, h.text, target_tokens=self.summary_tokens)
                h.metadata = dict(h.metadata)  # ensure not shared
                h.metadata["summary"] = summary

        return candidates
