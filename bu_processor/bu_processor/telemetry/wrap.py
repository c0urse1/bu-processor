from __future__ import annotations
from typing import List, Dict, Any, Optional
from bu_processor.telemetry.trace import Trace
from bu_processor.retrieval.models import RetrievalHit

def serialize_hits(hits: List[RetrievalHit]) -> List[Dict[str, Any]]:
    out = []
    for h in hits:
        out.append({
            "id": h.id,
            "score": float(h.score),
            "meta": {
                "doc_id": h.metadata.get("doc_id"),
                "section": h.metadata.get("section"),
                "page": h.metadata.get("page"),
                "title": h.metadata.get("title"),
            }
        })
    return out

def traced_retrieve(retriever, query: str, trace: Trace, **kwargs) -> List[RetrievalHit]:
    with trace.stage("retrieve", query=query, kwargs=kwargs):
        hits = retriever.retrieve(query, **kwargs)
    trace.event("retrieve.result", hits=serialize_hits(hits[:10]))  # cap for log size
    return hits

def traced_fuse(fuser, lists: List[List[RetrievalHit]], trace: Trace, name: str = "fuse.rrf"):
    with trace.stage(name, sizes=[len(x) for x in lists]):
        fused = fuser(lists)
    trace.event(f"{name}.result", hits=serialize_hits(fused[:10]))
    return fused

def traced_pack(packer, hits: List[RetrievalHit], trace: Trace, **kwargs):
    with trace.stage("context.pack", kwargs=kwargs):
        context_str, sources_table = packer(hits, **kwargs)
    trace.event("context.pack.result",
                ctx_chars=len(context_str), sources=len(sources_table),
                sources_table=sources_table[:10])  # safe meta only
    return context_str, sources_table
