# bu_processor/retrieval/dense.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

from bu_processor.ports import EmbeddingsBackend, VectorIndex
from bu_processor.storage.sqlite_store import SQLiteStore
from bu_processor.retrieval.models import RetrievalHit
from bu_processor.retrieval.filters import metadata_match

class DenseKnnRetriever:
    """
    Uses your VectorIndex (FAISS/Pinecone) + EmbeddingsBackend.
    """
    def __init__(self, embedder: EmbeddingsBackend, index: VectorIndex, store: SQLiteStore,
                 namespace: Optional[str] = None):
        self.embedder = embedder
        self.index = index
        self.store = store
        self.namespace = namespace

    def retrieve(self, query: str, top_k: int = 5,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[RetrievalHit]:
        qv = self.embedder.encode([query])[0]
        # Ask index to filter if it can; we still re-check locally for safety
        raw = self.index.query(qv, top_k=top_k, namespace=self.namespace, metadata_filter=metadata_filter)
        hits: List[RetrievalHit] = []
        for h in raw:
            cid = h["id"]
            row = self.store.get_chunk(cid)
            if not row:
                continue
            md = dict(h.get("metadata") or {})
            if not metadata_match(md, metadata_filter):
                continue
            hits.append(RetrievalHit(id=cid, score=float(h["score"]), text=row["text"], metadata=md))
        return hits
