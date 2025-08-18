from __future__ import annotations
from typing import List, Dict, Any, Optional
from statistics import mean

def hit_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    top = set(retrieved_ids[:k])
    return 1.0 if any(g in top for g in gold_ids) else 0.0

def reciprocal_rank(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    seen = set(gold_ids)
    for i, _id in enumerate(retrieved_ids, start=1):
        if _id in seen:
            return 1.0 / i
    return 0.0

def aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    agg = {}
    for key in ("hit@1", "hit@3", "hit@5", "mrr", "faithfulness", "citation_acc"):
        vals = [r[key] for r in rows if key in r]
        agg[key] = mean(vals) if vals else 0.0
    return agg

# -------- Answer-level sanity checks (deterministic heuristics) --------

def citation_accuracy(answer_text: str, sources_table: List[Dict[str, Any]]) -> float:
    """
    Checks that every paragraph ends with [...], and indices refer to existing sources.
    Score = (# valid paragraphs) / (# paragraphs)
    """
    paras = [p.strip() for p in answer_text.split("\n\n") if p.strip()]
    if not paras:
        return 0.0
    import re
    ok = 0
    for p in paras:
        m = re.search(r"\[(\d+(?:\s*,\s*\d+)*)]$", p)
        if not m:
            continue
        idxs = [int(x.strip()) for x in m.group(1).split(",") if x.strip().isdigit()]
        if idxs and all(1 <= i <= len(sources_table) for i in idxs):
            ok += 1
    return ok / max(1, len(paras))

def faithfulness_keywords(answer_text: str, cited_texts: List[str], keywords: List[str]) -> float:
    """
    Heuristic: if you provide a few gold keywords, check they occur in the answer AND at least
    one appears in the concatenated cited texts. Score ∈ {0, 0.5, 1}.
    """
    if not keywords:
        return 1.0
    ans = answer_text.lower()
    concat = " ".join(cited_texts).lower()
    in_ans = any(k.lower() in ans for k in keywords)
    in_ctx = any(k.lower() in concat for k in keywords)
    if in_ans and in_ctx:
        return 1.0
    if in_ans or in_ctx:
        return 0.5
    return 0.0
