from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
import numpy as np
from uuid import uuid4
from datetime import datetime, timezone

from bu_processor.ports import EmbeddingsBackend, VectorIndex
from bu_processor.storage.sqlite_store import SQLiteStore

# Import DocumentChunk for conversion
try:
    from ..models.chunk import DocumentChunk
except ImportError:
    DocumentChunk = None

def convert_chunks_to_dict(chunks: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert chunks to the dict format expected by embed_and_index_chunks.
    
    Handles both legacy dict format and new DocumentChunk objects.
    """
    result = []
    
    for chunk in chunks:
        if isinstance(chunk, dict):
            # Already in dict format
            result.append(chunk)
        elif DocumentChunk and hasattr(chunk, 'text'):
            # DocumentChunk object - convert to dict
            chunk_dict = {
                "text": chunk.text,
                "page": chunk.page_start if hasattr(chunk, 'page_start') else None,
                "section": chunk.section if hasattr(chunk, 'section') else None,
                "meta": {
                    # Rich metadata from DocumentChunk
                    "chunk_id": getattr(chunk, 'chunk_id', getattr(chunk, 'id', None)),
                    "doc_id": getattr(chunk, 'doc_id', None),
                    "page_start": getattr(chunk, 'page_start', None),
                    "page_end": getattr(chunk, 'page_end', None),
                    "heading_path": getattr(chunk, 'heading_path', None),
                    "heading_text": getattr(chunk, 'heading_text', None),
                    "chunk_type": getattr(chunk, 'chunk_type', None),
                    "importance_score": getattr(chunk, 'importance_score', None),
                    "char_span": getattr(chunk, 'char_span', None),
                    "start_position": getattr(chunk, 'start_position', None),
                    "end_position": getattr(chunk, 'end_position', None),
                    # Include original meta if present
                    **(getattr(chunk, 'meta', {}) or {})
                }
            }
            result.append(chunk_dict)
        else:
            # Fallback: try to extract text
            result.append({
                "text": str(chunk),
                "page": None,
                "section": None,
                "meta": {}
            })
    
    return result

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
    doc_id: Optional[str] = None,  # NEW: Allow pre-generated doc_id
) -> Dict[str, Any]:
    """
    1) Store document + chunks in SQLite (get stable chunk_ids)
    2) Embed chunk texts
    3) Upsert vectors into VectorIndex with (id, vector, metadata)
    
    NEW: If doc_id is provided, use it instead of generating a new one.
    This enables stable doc_ids and proper metadata flow.
    B3: Adds ingestion timestamps to document and chunk metadata.
    """
    # B3) Generate ingestion timestamp
    ingested_at = datetime.now(timezone.utc).isoformat()
    
    # 1) persist doc + chunks with stable doc_id and timestamps
    enhanced_doc_meta = dict(doc_meta or {})
    enhanced_doc_meta["ingested_at"] = ingested_at
    
    final_doc_id = store.upsert_document(doc_id=doc_id, title=doc_title, source=doc_source, meta=enhanced_doc_meta)
    # Enhance chunks with timestamps before storing
    enhanced_chunks = []
    for c in chunks:
        enhanced_chunk = dict(c)
        if "meta" not in enhanced_chunk:
            enhanced_chunk["meta"] = {}
        enhanced_chunk["meta"]["ingested_at"] = ingested_at
        enhanced_chunks.append(enhanced_chunk)
    
    chunk_ids = store.upsert_chunks(final_doc_id, [{"text": c["text"], "page": c.get("page"),
                                              "section": c.get("section"), "meta": c.get("meta")} for c in enhanced_chunks])

    # 2) embed
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts)  # shape [N, D]
    dim = vectors.shape[1]
    index.create(dim=dim, metric="cosine", namespace=namespace)

    # 3) upsert to index with rich metadata - B2) Enhanced metadata flow + B3) Timestamps
    metadata = []
    for cid, c in zip(chunk_ids, enhanced_chunks):
        # Extract rich metadata from chunk (B2 implementation)
        chunk_meta = c.get("meta", {}) or {}
        
        md = {
            "doc_id": final_doc_id,
            "chunk_id": cid,
            "title": doc_title,
            "source": doc_source,
            
            # B3) Ingestion timestamp
            "ingested_at": ingested_at,
            
            # B2) New rich metadata fields
            "page_start": chunk_meta.get("page_start") or c.get("page"),  # Fallback to legacy "page"
            "page_end": chunk_meta.get("page_end") or c.get("page"),
            "section": c.get("section"),
            "heading_path": chunk_meta.get("heading_path"),
            
            # Additional semantic chunking metadata
            "chunk_type": chunk_meta.get("chunk_type"),
            "importance_score": chunk_meta.get("importance_score"),
            "heading_text": chunk_meta.get("heading_text"),
            "start_position": chunk_meta.get("start_position"),
            "end_position": chunk_meta.get("end_position"),
            
            # Include all other metadata from chunk.meta
            **chunk_meta,
        }
        metadata.append(md)

    index.upsert(ids=chunk_ids, vectors=vectors, metadata=metadata, namespace=namespace)
    return {"doc_id": final_doc_id, "chunk_ids": chunk_ids, "dim": dim, "ingested_at": ingested_at}
