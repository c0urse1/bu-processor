# bu_processor/retrieval/fusion.py
from __future__ import annotations
from typing import List, Dict
from bu_processor.retrieval.models import RetrievalHit

def rrf_fuse(lists: List[List[RetrievalHit]], k: int = 60) -> List[RetrievalHit]:
    """
    Reciprocal Rank Fusion: score = sum(1 / (k + rank))
    Returns a deduped, fused list with summed RRF scores.
    """
    score_map: Dict[str, float] = {}
    any_meta: Dict[str, Dict] = {}
    any_text: Dict[str, str] = {}

    for lst in lists:
        for rank, h in enumerate(lst, start=1):
            score_map[h.id] = score_map.get(h.id, 0.0) + 1.0 / (k + rank)
            any_meta.setdefault(h.id, h.metadata)
            any_text.setdefault(h.id, h.text)

    fused = [RetrievalHit(id=_id, score=s, text=any_text[_id], metadata=any_meta[_id])
             for _id, s in score_map.items()]
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused

def weighted_sum_fuse(dense: List[RetrievalHit], bm25: List[RetrievalHit],
                      alpha_dense: float = 0.5, alpha_bm25: float = 0.5) -> List[RetrievalHit]:
    """
    Normalize each list to [0,1] then blend: alpha_dense * s_dense + alpha_bm25 * s_bm25
    Missing ids get treated as 0 in the other list.
    """
    def norm(lst: List[RetrievalHit]) -> Dict[str, float]:
        if not lst:
            return {}
        scores = [h.score for h in lst]
        smin, smax = min(scores), max(scores)
        rng = (smax - smin) or 1e-8
        return {h.id: (h.score - smin) / rng for h in lst}

    nd, nb = norm(dense), norm(bm25)
    ids = set(nd) | set(nb)
    id2meta = {h.id: h.metadata for h in dense + bm25}
    id2text = {h.id: h.text for h in dense + bm25}

    fused = []
    for _id in ids:
        s = alpha_dense * nd.get(_id, 0.0) + alpha_bm25 * nb.get(_id, 0.0)
        fused.append(RetrievalHit(id=_id, score=s, text=id2text[_id], metadata=id2meta[_id]))
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused
