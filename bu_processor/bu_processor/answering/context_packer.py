from __future__ import annotations
from typing import List, Dict, Any, Tuple, Set
import re
from bu_processor.retrieval.models import RetrievalHit
from bu_processor.semantic.tokens import approx_token_count

def _sent_split(text: str) -> List[str]:
    # lightweight sentence splitter; replace with better if you have one
    return re.split(r'(?<=[.!?])\s+', text.strip())

def _normalize(s: str) -> str:
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

def _anti_dup_sentences(sentences: List[str], seen: Set[str]) -> List[str]:
    out = []
    for s in sentences:
        key = _normalize(s)
        if len(key) < 3:  # skip empties
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

def build_sources_table(hits: List[RetrievalHit]) -> List[Dict[str, Any]]:
    """
    Make a stable table mapping numeric [i] -> source metadata.
    Order is the order provided by caller (usually already reranked).
    """
    table = []
    for h in hits:
        table.append({
            "chunk_id": h.id,
            "doc_id": h.metadata.get("doc_id") or "",
            "title": h.metadata.get("title") or "",
            "section": h.metadata.get("section") or "",
            "page": h.metadata.get("page"),
            "score": h.score,
        })
    return table

def pack_context(
    hits: List[RetrievalHit],
    token_budget: int = 1200,
    sentence_overlap: int = 1,
    prefer_summary: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Pack top chunks into a single context string with headings per source:
    - anti-duplicate sentences across all chunks
    - enforce a global token budget
    - include small overlap between chunks for continuity
    - prefer summarizer output if available in hit.metadata["summary"]
    Returns (context_str, sources_table) where sources_table aligns with [1], [2], ...
    """
    seen: Set[str] = set()
    total_tokens = 0
    used_hits: List[RetrievalHit] = []

    for h in hits:
        # choose text or summarizer result
        text = (h.metadata.get("summary") or h.text) if prefer_summary else h.text
        sents = _sent_split(text)

        # keep up to N sentences, dedup across all
        sents = _anti_dup_sentences(sents, seen)
        if not sents:
            continue

        # add overlap with previous chunk
        if used_hits and sentence_overlap > 0:
            prev = (used_hits[-1].metadata.get("summary") or used_hits[-1].text) if prefer_summary else used_hits[-1].text
            prev_sents = _sent_split(prev)
            tail = prev_sents[-sentence_overlap:]
            head = sents[:max(1, sentence_overlap)]
            sents = tail + head + sents  # tiny redundancy to help flow

        candidate = " ".join(sents)
        cand_tokens = approx_token_count(candidate)
        if total_tokens + cand_tokens > token_budget:
            # try a reduced portion
            trimmed = []
            for s in sents:
                t = approx_token_count(s)
                if total_tokens + t > token_budget:
                    break
                trimmed.append(s)
                total_tokens += t
            if trimmed:
                # keep partially
                used_hits.append(h)
            break
        else:
            used_hits.append(h)
            total_tokens += cand_tokens

    # Build numbered sources and stitched context
    sources_table = build_sources_table(used_hits)
    parts = []
    for idx, h in enumerate(used_hits, start=1):
        header = f"[{idx}] {h.metadata.get('title') or h.metadata.get('section') or 'Source'}"
        meta = []
        if h.metadata.get("doc_id"):   meta.append(f"doc:{h.metadata['doc_id']}")
        if h.metadata.get("section"):  meta.append(f"sec:{h.metadata['section']}")
        if h.metadata.get("page") is not None: meta.append(f"p.{h.metadata['page']}")
        header_meta = f" ({', '.join(meta)})" if meta else ""
        body = (h.metadata.get("summary") or h.text) if prefer_summary else h.text
        parts.append(f"{header}{header_meta}\n{body}\n")
    context_str = "\n".join(parts).strip()
    return context_str, sources_table
