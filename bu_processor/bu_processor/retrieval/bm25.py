# bu_processor/retrieval/bm25.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

from rank_bm25 import BM25Okapi

from bu_processor.storage.sqlite_store import SQLiteStore
from bu_processor.retrieval.models import RetrievalHit
from bu_processor.retrieval.filters import metadata_match

_WORD = re.compile(r"\w+", flags=re.UNICODE)

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text)]

@dataclass
class _Doc:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

class Bm25Index:
    """
    Simple in-memory BM25 index over chunks stored in SQLite.
    Build once, refresh as needed (for CI-sized corpora).
    """
    def __init__(self, store: SQLiteStore):
        self.store = store
        self.docs: List[_Doc] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def build_from_store(self, where: Optional[Dict[str, Any]] = None) -> None:
        """
        Load all chunks; 'where' can filter by doc_id, etc. (extend as needed).
        """
        # Minimal store API didn't include listing; we infer a small helper here:
        # For production, add a proper list method to SQLiteStore.
        # Temporary approach: introspect raw DB (works for tests).
        from sqlalchemy import select
        with self.store.engine.begin() as conn:
            rows = conn.execute(select(self.store.chunks)).fetchall()
        self.docs = []
        self.corpus_tokens = []
        for r in rows:
            md = dict(r.meta or {})
            md.update({"doc_id": r.doc_id, "chunk_id": r.chunk_id, "page": r.page, "section": r.section})
            d = _Doc(chunk_id=r.chunk_id, text=r.text, metadata=md)
            self.docs.append(d)
            self.corpus_tokens.append(_tokenize(d.text))
        
        # Only build BM25 if we have documents
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def query(self, query: str, top_k: int = 5,
              metadata_filter: Optional[Dict[str, Any]] = None) -> List[RetrievalHit]:
        if not self.bm25 or not self.docs:
            return []
        q_tokens = _tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        # get top_k indices (descending)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: List[RetrievalHit] = []
        for i in idxs:
            if len(hits) >= top_k:
                break
            doc = self.docs[i]
            md = doc.metadata
            if not metadata_match(md, metadata_filter):
                continue
            hits.append(RetrievalHit(id=doc.chunk_id, score=float(scores[i]), text=doc.text, metadata=md))
        return hits


# Note: For production scale, add a list_chunks() method to SQLiteStore so you don't reach into SQLAlchemy directly. For tests and small corpora this is fine.
