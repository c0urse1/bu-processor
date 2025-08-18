from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from bu_processor.ports import EmbeddingsBackend, VectorIndex
from bu_processor.storage.sqlite_store import SQLiteStore

def retrieve_similar(
    *,
    query: str,
    embedder: EmbeddingsBackend,
    index: VectorIndex,
    store: SQLiteStore,
    top_k: int = 5,
    namespace: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    qv = embedder.encode([query])[0]
    hits = index.query(qv, top_k=top_k, namespace=namespace, metadata_filter=metadata_filter)
    # attach text for synthesis/citations
    out = []
    for h in hits:
        cid = h["id"]
        row = store.get_chunk(cid)
        if row:
            h["text"] = row["text"]
        out.append(h)
    return out
