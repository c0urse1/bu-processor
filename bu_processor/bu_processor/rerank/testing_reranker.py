from __future__ import annotations
from typing import List, Optional
import math
from bu_processor.retrieval.models import RetrievalHit

class HeuristicOverlapReranker:
    """
    Deterministic, no-network reranker for tests.
    Scores by token overlap and term proximity; favors presence of rare query tokens.
    """
    def __init__(self):
        pass

    def _score(self, query: str, text: str) -> float:
        q = [t for t in query.lower().split() if t.isalpha()]
        t = text.lower()
        if not q or not t:
            return 0.0
        score = 0.0
        for term in q:
            cnt = t.count(term)
            if cnt:
                score += 2.0 * math.log1p(cnt)  # diminishing returns
        # slight length normalization
        L = max(20, len(text))
        score /= math.log(L)
        return score

    def rerank(self, query: str, hits: List[RetrievalHit], top_k: Optional[int] = None) -> List[RetrievalHit]:
        ranked = sorted(hits, key=lambda h: self._score(query, h.text), reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        # annotate for traceability
        for i, h in enumerate(ranked):
            h.metadata = dict(h.metadata)
            h.metadata["ce_score"] = float(self._score(query, h.text))
        return ranked
