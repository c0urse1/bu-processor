# bu_processor/retrieval/mmr.py
from __future__ import annotations
from typing import List, Sequence
import numpy as np
from bu_processor.retrieval.models import RetrievalHit

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-8)

def mmr_select(
    query_vec: np.ndarray,
    candidates: List[RetrievalHit],
    get_vec: callable,
    top_k: int = 5,
    lambda_mult: float = 0.65,
) -> List[RetrievalHit]:
    """
    Greedy MMR: select documents maximizing
        λ * sim(query, doc) - (1 - λ) * max_{d' in selected} sim(doc, d')
    where sim is cosine on normalized vectors.
    `get_vec(hit)` must return a np.ndarray vector for the hit's text.
    """
    if not candidates:
        return []
    q = _normalize(query_vec)
    selected: List[RetrievalHit] = []
    cand_vecs = [ _normalize(get_vec(h)) for h in candidates ]

    scores_to_q = [ _cos(q, v) for v in cand_vecs ]
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_i, best_score = None, -1e9
        for i in remaining:
            # diversity penalty
            max_sim_to_sel = 0.0
            if selected:
                for s_idx, s in enumerate(selected):
                    sim = _cos(cand_vecs[i], _normalize(get_vec(s)))
                    if sim > max_sim_to_sel:
                        max_sim_to_sel = sim
            score = lambda_mult * scores_to_q[i] - (1.0 - lambda_mult) * max_sim_to_sel
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(candidates[best_i])
        remaining.remove(best_i)
    return selected
