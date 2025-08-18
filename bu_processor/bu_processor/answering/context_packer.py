from __future__ import annotations
from typing import List, Dict, Any, Tuple, Set
import math, re
from statistics import mean

from bu_processor.retrieval.models import RetrievalHit
from bu_processor.semantic.tokens import approx_token_count

NORM_WS = re.compile(r"\s+")
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _norm(s: str) -> str:
    return NORM_WS.sub(" ", s.strip()).lower()

def _split_sents(text: str) -> List[str]:
    return [t.strip() for t in SENT_SPLIT.split(text.strip()) if t.strip()]

def _anti_dup_sentences(sents: List[str], seen: Set[str]) -> List[str]:
    out = []
    for s in sents:
        k = _norm(s)
        if len(k) < 3 or k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out

def _build_sources_table(hits: List[RetrievalHit]) -> List[Dict[str, Any]]:
    table = []
    for h in hits:
        table.append({
            "chunk_id": h.id,
            "doc_id": h.metadata.get("doc_id") or "",
            "title": h.metadata.get("title") or "",
            "section": h.metadata.get("section") or "",
            "page_start": h.metadata.get("page_start"),
            "page_end": h.metadata.get("page_end"),
            "score": float(h.score),
        })
    return table

def _score_quota(scores: List[float]) -> List[float]:
    if not scores:
        return []
    # normalize to [0,1]; softmax-ish to avoid a single source dominating
    mn, mx = min(scores), max(scores)
    rng = (mx - mn) or 1e-6
    norm = [(s - mn) / rng for s in scores]
    # temperature 0.7
    norm = [math.pow(x, 0.7) for x in norm]
    s = sum(norm) or 1.0
    return [x / s for x in norm]

def pack_context(
    hits: List[RetrievalHit],
    token_budget: int = 1200,
    sentence_overlap: int = 1,
    prefer_summary: bool = True,
    per_source_min_tokens: int = 80,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    1) Select top candidates (unique by chunk_id)
    2) Allocate budget proportional to scores (quota)
    3) From each source, take sentences (summary or text), add tiny overlap, anti-dup
    4) Build numbered sections [i] ... per source
    """
    # 1) uniq by chunk_id, keep order
    uniq = []
    seen_ids = set()
    for h in hits:
        if h.id not in seen_ids:
            uniq.append(h); seen_ids.add(h.id)

    if not uniq:
        return "", []

    # 2) score quotas
    scores = [float(h.score) for h in uniq]
    quotas = _score_quota(scores)
    budgets = [max(per_source_min_tokens, int(q * token_budget)) for q in quotas]

    # 3) collect text with anti-dup
    seen_sents: Set[str] = set()
    pieces: List[Tuple[int, str, Dict[str, Any]]] = []  # (source_index, text, meta)
    total_tokens = 0

    for idx, (h, budget) in enumerate(zip(uniq, budgets), start=1):
        base = (h.metadata.get("summary") or h.text) if prefer_summary else h.text
        sents = _split_sents(base)
        sents = _anti_dup_sentences(sents, seen_sents)
        if not sents:
            continue

        # add tiny overlap with previous picked source (coherence)
        if pieces and sentence_overlap > 0:
            prev_text = pieces[-1][1]
            prev_last = _split_sents(prev_text)[-sentence_overlap:]
            head = sents[:max(1, sentence_overlap)]
            sents = prev_last + head + sents

        # take sentences until budget
        taken = []
        used = 0
        for s in sents:
            t = approx_token_count(s)
            if used + t > budget:
                break
            taken.append(s); used += t

        if not taken:
            continue

        body = " ".join(taken).strip()
        meta = {
            "title": h.metadata.get("title") or h.metadata.get("section") or "Source",
            "doc_id": h.metadata.get("doc_id"),
            "section": h.metadata.get("section"),
            "page_start": h.metadata.get("page_start") or h.metadata.get("page"),
            "page_end": h.metadata.get("page_end") or h.metadata.get("page"),
        }
        pieces.append((idx, body, meta))
        total_tokens += approx_token_count(body)
        if total_tokens >= token_budget:
            break

    # 4) sources and stitched context
    used_hits = [uniq[i-1] for (i,_,_) in pieces]
    sources_table = _build_sources_table(used_hits)

    blocks = []
    for (i, body, meta) in pieces:
        header = f"[{i}] {meta['title']}"
        trail = []
        if meta.get("doc_id"): trail.append(f"doc:{meta['doc_id']}")
        if meta.get("section"): trail.append(f"sec:{meta['section']}")
        if meta.get("page_start") is not None:
            if meta.get("page_end") and meta["page_end"] != meta["page_start"]:
                trail.append(f"p.{meta['page_start']}-{meta['page_end']}")
            else:
                trail.append(f"p.{meta['page_start']}")
        suffix = (" (" + ", ".join(trail) + ")") if trail else ""
        blocks.append(header + suffix + "\n" + body)
    return "\n\n".join(blocks).strip(), sources_table
