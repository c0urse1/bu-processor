# bu_processor/integrations/pinecone_enhanced.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os

# Flags kannst du später nutzen, um Features zuzuschalten
from bu_processor.core.flags import (
    ENABLE_METRICS, ENABLE_RATE_LIMITER, ENABLE_EMBED_CACHE, ENABLE_RERANK
)

# Wir nutzen die Simple-Implementierung als inneren Motor
from .pinecone_simple import PineconeManager as _SimplePinecone

# (Optional) Metrik-Adapter – macht bei ENABLE_METRICS=False nichts
try:
    from bu_processor.observability.metrics import upsert_latency  # HistogramClass
except Exception:
    upsert_latency = None

def _maybe_observe(hist, seconds: float) -> None:
    if hist is not None and hasattr(hist, "observe"):
        try:
            hist.observe(seconds)
        except Exception:
            pass

class PineconeEnhancedManager:
    """
    Enhanced-Variante als dünne Fassade um den Simple-Manager.
    - bietet dieselbe API wie Simple
    - Platz für spätere Features: Async-Upserts, Retry, Rate-Limits, Prometheus, Rerank
    - aktuell synchron & delegiert an _SimplePinecone
    """

    def __init__(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,  # v2
        cloud: Optional[str] = None,        # v3
        region: Optional[str] = None,       # v3
        metric: str = "cosine",
        namespace: Optional[str] = None
    ):
        # Kein NotImplementedError mehr – wir sind lauffähig.
        self._inner = _SimplePinecone(
            index_name=index_name,
            api_key=api_key,
            environment=environment,
            cloud=cloud,
            region=region,
            metric=metric,
            namespace=namespace,
        )

    # --- Management ---
    def ensure_index(self, dimension: int) -> None:
        self._inner.ensure_index(dimension)

    def get_index_dimension(self) -> Optional[int]:
        return self._inner.get_index_dimension()

    # --- Upsert ---
    def upsert_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None
    ) -> Any:
        # Hook für Metrics/Rate-Limit/Retry – später zuschalten
        if ENABLE_METRICS and upsert_latency is not None:
            import time
            t0 = time.time()
            try:
                return self._inner.upsert_vectors(ids, vectors, metadatas, namespace)
            finally:
                _maybe_observe(upsert_latency, time.time() - t0)
        return self._inner.upsert_vectors(ids, vectors, metadatas, namespace)

    def upsert_items(
        self,
        items: List[Dict[str, Any]],
        namespace: Optional[str] = None
    ) -> Any:
        return self._inner.upsert_items(items, namespace)

    # --- Query ---
    def query_by_vector(
        self,
        vector: List[float],
        top_k: int = 5,
        include_metadata: bool = True,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        res = self._inner.query_by_vector(
            vector, top_k=top_k, include_metadata=include_metadata,
            namespace=namespace, filter=filter
        )
        # Platz für Reranking via Cross-Encoder
        # if ENABLE_RERANK: res = self._rerank(res, vector_or_text=...)
        return res

    def query_by_text(
        self,
        text: str,
        embedder,
        top_k: int = 5,
        include_metadata: bool = True,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        res = self._inner.query_by_text(
            text, embedder, top_k=top_k, include_metadata=include_metadata,
            namespace=namespace, filter=filter
        )
        # if ENABLE_RERANK: res = self._rerank(res, query_text=text)
        return res

    # --- Delete ---
    def delete_by_document_id(self, doc_id: str, namespace: Optional[str] = None) -> Any:
        return self._inner.delete_by_document_id(doc_id, namespace)

    # --- (Optional) Rerank-Skizze für später ---
    # def _rerank(self, pinecone_result: Dict[str, Any], query_text: str | None = None, vector_or_text=None):
    #     # Hier könntest du mit cross-encoder/ms-marco-MiniLM-L-6-v2 Scores nachziehen und neu sortieren.
    #     return pinecone_result
