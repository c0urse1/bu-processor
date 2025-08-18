from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

from bu_processor.ports import EmbeddingsBackend, VectorIndex
from bu_processor.storage.sqlite_store import SQLiteStore

def embed_and_index_chunks(
    *,
    doc_title: Optional[str],
    doc_source: Optional[str],
    doc_meta: Optional[Dict[str, Any]],
    chunks: List[Dict[str, Any]],   # each: {"text": str, "page": int?, "section": str?, "meta": dict?}
    embedder: EmbeddingsBackend,
    index: VectorIndex,
    store: SQLiteStore,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """
    1) Store document + chunks in SQLite (get stable chunk_ids)
    2) Embed chunk texts
    3) Upsert vectors into VectorIndex with (id, vector, metadata)
    """
    # 1) persist doc + chunks
    doc_id = store.upsert_document(doc_id=None, title=doc_title, source=doc_source, meta=doc_meta)
    chunk_ids = store.upsert_chunks(doc_id, [{"text": c["text"], "page": c.get("page"),
                                              "section": c.get("section"), "meta": c.get("meta")} for c in chunks])

    # 2) embed
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts)  # shape [N, D]
    dim = vectors.shape[1]
    index.create(dim=dim, metric="cosine", namespace=namespace)

    # 3) upsert to index with rich metadata
    metadata = []
    for cid, c in zip(chunk_ids, chunks):
        md = {
            "doc_id": doc_id,
            "chunk_id": cid,
            "title": doc_title,
            "source": doc_source,
            "page": c.get("page"),
            "section": c.get("section"),
            **(c.get("meta") or {}),
        }
        metadata.append(md)

    index.upsert(ids=chunk_ids, vectors=vectors, metadata=metadata, namespace=namespace)
    return {"doc_id": doc_id, "chunk_ids": chunk_ids, "dim": dim}
