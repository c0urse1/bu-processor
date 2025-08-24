# bu_processor/integrations/pinecone_simple.py
from __future__ import annotations
import os
from typing import List, Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..embeddings.embedder import Embedder

# Import observability (No-Op if flags disabled)
try:
    from ..observability.metrics import (
        pinecone_operations, pinecone_latency, time_operation
    )
    from ..core.ratelimit import pinecone_limiter, rate_limited_operation
    _OBSERVABILITY_AVAILABLE = True
except ImportError:
    # Fallback if observability not available
    _OBSERVABILITY_AVAILABLE = False

# Import quality gates and reranking
try:
    from ..core.quality_gates import apply_quality_gates, QualityGateError
    from ..core.reranking import rerank_search_results
    from ..core.flags import ENABLE_RERANK
    _QUALITY_GATES_AVAILABLE = True
except ImportError:
    # Fallback if quality gates not available
    _QUALITY_GATES_AVAILABLE = False
    ENABLE_RERANK = False

# v3?
try:
    from pinecone import Pinecone, ServerlessSpec
    _PC_V3 = True
except Exception:
    _PC_V3 = False

# v2 fallback?
if not _PC_V3:
    try:
        import pinecone as pc_v2  # type: ignore
    except ImportError:
        pc_v2 = None

class PineconeManager:
    """
    Minimaler, stabiler MVP-Wrapper für Pinecone (v2 oder v3).
    
    Fokus auf Einfachheit und Stabilität für MVP.
    Unterstützt:
      - ensure_index(dimension)
      - upsert_vectors(ids, vectors, metadatas, namespace)
      - upsert_items(items, namespace)
      - query_by_vector(vector, top_k, filter, namespace)
      - query_by_text(text, embedder, ...)
      - get_index_dimension()
      - delete_by_document_id(doc_id, namespace)
    """
    def __init__(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,     # v2
        cloud: Optional[str] = None,           # v3 serverless (z.B. "gcp")
        region: Optional[str] = None,          # v3 serverless (z.B. "us-west1")
        metric: str = "cosine",
        namespace: Optional[str] = None
    ):
        self.index_name = index_name
        self.metric = metric
        self.namespace = namespace

        api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY fehlt")

        if _PC_V3:
            self.client = Pinecone(api_key=api_key)
            self.cloud = cloud or os.getenv("PINECONE_CLOUD", "gcp")
            self.region = region or os.getenv("PINECONE_REGION", "us-west1")
            # Index-Objekt wird erst nach ensure_index erstellt
            self.index = None
        else:
            if pc_v2 is None:
                raise RuntimeError("Weder Pinecone v3 noch v2 verfügbar")
            env = environment or os.getenv("PINECONE_ENV", "us-west1-gcp")
            pc_v2.init(api_key=api_key, environment=env)
            self.client = None
            self.index = None  # wird nach ensure_index gesetzt

    # --- Management ---
    def ensure_index(self, dimension: int) -> None:
        """Stelle sicher, dass der Index existiert."""
        if _PC_V3:
            names = {ix.name for ix in self.client.list_indexes()}
            if self.index_name not in names:
                self.client.create_index(
                    name=self.index_name,
                    dimension=int(dimension),
                    metric=self.metric,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )
            self.index = self.client.Index(self.index_name)
        else:
            # v2
            names = set(pc_v2.list_indexes())
            if self.index_name not in names:
                pc_v2.create_index(
                    name=self.index_name,
                    dimension=int(dimension),
                    metric=self.metric
                )
            self.index = pc_v2.Index(self.index_name)

    def get_index_dimension(self) -> Optional[int]:
        """Hole die Dimension des Index."""
        try:
            if _PC_V3:
                desc = self.client.describe_index(self.index_name)
                return int(desc.dimension)  # type: ignore
            else:
                # v2: kein einheitlicher Weg; versuche describe_index_stats
                stats = self.index.describe_index_stats()  # type: ignore
                # dimension ist nicht garantiert vorhanden -> best effort
                return int(stats.get("dimension")) if "dimension" in stats else None
        except Exception:
            return None

    # --- Upsert with Quality Gates ---
    def upsert_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None,
        embedder: Optional["Embedder"] = None,
        skip_quality_gates: bool = False
    ) -> Any:
        """
        Lade Vektoren in den Index hoch mit Qualitätsgatter.
        
        Args:
            ids: Vector IDs
            vectors: Vector embeddings  
            metadatas: Metadata dictionaries
            namespace: Optional namespace
            embedder: Embedder instance for dimension checks
            skip_quality_gates: Skip quality gates (for testing)
        """
        # Quality Gate: Apply consistency checks before upsert
        if not skip_quality_gates and _QUALITY_GATES_AVAILABLE and embedder:
            try:
                apply_quality_gates(
                    pinecone_manager=self,
                    embedder=embedder,
                    ids=ids,
                    vectors=vectors,
                    metadatas=metadatas
                )
            except QualityGateError as e:
                raise RuntimeError(f"Quality gate failed: {e}") from e
        
        if metadatas is None:
            metadatas = [{} for _ in ids]
        items = [{"id": i, "values": v, "metadata": m}
                 for i, v, m in zip(ids, vectors, metadatas)]
        return self.upsert_items(items=items, namespace=namespace, skip_quality_gates=True)

    def upsert_items(
        self,
        items: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        embedder: Optional["Embedder"] = None,
        skip_quality_gates: bool = False
    ) -> Any:
        """
        Lade Items in den Index hoch mit Qualitätsgatter.
        
        Args:
            items: Item dictionaries with id, values, metadata
            namespace: Optional namespace
            embedder: Embedder instance for dimension checks
            skip_quality_gates: Skip quality gates (for testing)
        """
        # Quality Gate: Apply consistency checks before upsert
        if not skip_quality_gates and _QUALITY_GATES_AVAILABLE and embedder:
            try:
                apply_quality_gates(
                    pinecone_manager=self,
                    embedder=embedder,
                    items=items
                )
            except QualityGateError as e:
                raise RuntimeError(f"Quality gate failed: {e}") from e
        
        ns = namespace or self.namespace
        
        # Observability integration (No-Op if flags disabled)
        if _OBSERVABILITY_AVAILABLE:
            with rate_limited_operation(pinecone_limiter, "upsert_items"):
                with time_operation(pinecone_latency, operation="upsert"):
                    try:
                        if _PC_V3:
                            result = self.index.upsert(vectors=items, namespace=ns)  # type: ignore
                        else:
                            result = self.index.upsert(vectors=items, namespace=ns)  # type: ignore
                        
                        # Track successful operation
                        pinecone_operations.labels(operation="upsert", status="success").inc()
                        return result
                        
                    except Exception as e:
                        # Track failed operation
                        pinecone_operations.labels(operation="upsert", status="error").inc()
                        raise
        else:
            # Fallback without observability
            if _PC_V3:
                return self.index.upsert(vectors=items, namespace=ns)  # type: ignore
            else:
                return self.index.upsert(vectors=items, namespace=ns)  # type: ignore

    # --- Query ---
    def query_by_vector(
        self,
        vector: List[float],
        top_k: int = 5,
        include_metadata: bool = True,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Suche ähnliche Vektoren."""
        ns = namespace or self.namespace
        flt = filter or {}
        return self.index.query(  # type: ignore
            vector=vector, top_k=top_k,
            include_metadata=include_metadata,
            namespace=ns, filter=flt
        )

    def query_by_text(
        self,
        text: str,
        embedder: "Embedder",
        top_k: int = 5,
        include_metadata: bool = True,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        enable_rerank: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Suche ähnliche Texte über Embedding mit optionalem Reranking.
        
        Args:
            text: Search query text
            embedder: Embedder instance
            top_k: Number of results to return
            include_metadata: Include metadata in results
            namespace: Optional namespace
            filter: Optional metadata filter
            enable_rerank: Override ENABLE_RERANK flag (None = use flag)
            
        Returns:
            Search results, optionally reranked
        """
        # Get initial results from Pinecone
        vec = embedder.encode_one(text)
        pinecone_results = self.query_by_vector(
            vec, top_k, include_metadata, namespace, filter
        )
        
        # Apply reranking if enabled
        should_rerank = enable_rerank if enable_rerank is not None else ENABLE_RERANK
        if should_rerank and _QUALITY_GATES_AVAILABLE and "matches" in pinecone_results:
            try:
                # Extract matches for reranking
                matches = pinecone_results.get("matches", [])
                if matches:
                    # Convert to format expected by reranker
                    results_for_rerank = []
                    for match in matches:
                        result = {
                            "id": match.get("id", ""),
                            "score": match.get("score", 0.0),
                            "metadata": match.get("metadata", {})
                        }
                        # Try to find text content in metadata
                        if "text" in result["metadata"]:
                            result["text"] = result["metadata"]["text"]
                        elif "content" in result["metadata"]:
                            result["text"] = result["metadata"]["content"]
                        else:
                            result["text"] = ""  # Fallback
                        results_for_rerank.append(result)
                    
                    # Apply reranking
                    reranked = rerank_search_results(
                        query=text,
                        results=results_for_rerank,
                        max_results=top_k
                    )
                    
                    # Convert back to Pinecone format
                    reranked_matches = []
                    for result in reranked:
                        match = {
                            "id": result["id"],
                            "score": result["score"],
                            "metadata": result["metadata"]
                        }
                        # Add reranking metadata
                        if "original_score" in result:
                            match["metadata"]["original_score"] = result["original_score"]
                        if "cross_encoder_score" in result:
                            match["metadata"]["cross_encoder_score"] = result["cross_encoder_score"]
                        reranked_matches.append(match)
                    
                    # Update results with reranked matches
                    pinecone_results["matches"] = reranked_matches
                    pinecone_results["reranked"] = True
                    
            except Exception as e:
                # Reranking failed, log and continue with original results
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Reranking failed, using original results: {e}")
        
        return pinecone_results

    # --- Delete ---
    def delete_by_document_id(self, doc_id: str, namespace: Optional[str] = None) -> Any:
        """Lösche Dokument anhand der doc_id."""
        ns = namespace or self.namespace
        return self.index.delete(  # type: ignore
            namespace=ns,
            filter={"doc_id": {"$eq": doc_id}}
        )

    # --- Legacy Compatibility ---
    def upsert_document(self, *, ids=None, vectors=None, metadatas=None, namespace=None, items=None):
        """Legacy wrapper for upsert functionality."""
        if items is not None:
            return self.upsert_items(items, namespace=namespace)
        return self.upsert_vectors(ids, vectors, metadatas, namespace)

    def search_similar_documents(
        self,
        query_vector: List[float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Legacy wrapper for search functionality."""
        result = self.query_by_vector(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            filter=filter_dict
        )
        
        # Convert to legacy format
        matches = []
        for match in result.get("matches", []):
            matches.append({
                "doc_id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {})
            })
        return matches

# Factory function für Kompatibilität
def get_pinecone_manager(**kwargs) -> PineconeManager:
    """Factory function to create PineconeManager instance."""
    return PineconeManager(**kwargs)

# Exports
__all__ = [
    "PineconeManager",
    "get_pinecone_manager"
]
