# bu_processor/integrations/pinecone_facade.py
"""
Pinecone integration facade.

This module provides a unified interface to Pinecone functionality,
automatically selecting between simple and enhanced implementations
based on feature flags and system capabilities.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Any, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..embeddings.embedder import Embedder

# Import feature flags
try:
    from ..core.flags import FeatureFlags
    flags = FeatureFlags()
except ImportError:
    # Fallback if flags not available
    class _FallbackFlags:
        enable_enhanced_pinecone = False
    flags = _FallbackFlags()

class PineconeManager:
    """
    Facade for Pinecone operations that automatically selects
    the appropriate implementation (simple or enhanced) based
    on feature flags and system configuration.
    """
    
    def __init__(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        metric: str = "cosine",
        namespace: Optional[str] = None,
        force_simple: bool = False
    ):
        """
        Initialize Pinecone manager.
        
        Args:
            index_name: Name of the Pinecone index
            api_key: Pinecone API key (defaults to environment variable)
            environment: Pinecone environment (v2 compatibility)
            cloud: Cloud provider for serverless (v3)
            region: Region for serverless (v3)
            metric: Distance metric for the index
            namespace: Default namespace for operations
            force_simple: Force use of simple implementation
        """
        self.index_name = index_name
        self.api_key = api_key
        self.environment = environment
        self.cloud = cloud
        self.region = region
        self.metric = metric
        self.namespace = namespace
        
        # Determine which implementation to use
        use_enhanced = (
            getattr(flags, 'enable_enhanced_pinecone', False) 
            and not force_simple
        )
        
        if use_enhanced:
            try:
                self._impl = self._create_enhanced_manager()
            except Exception:
                # Fall back to simple if enhanced fails
                self._impl = self._create_simple_manager()
        else:
            self._impl = self._create_simple_manager()
    
    def _create_simple_manager(self):
        """Create simple Pinecone manager instance."""
        from .pinecone_simple import PineconeManager as SimplePineconeManager
        return SimplePineconeManager(
            index_name=self.index_name,
            api_key=self.api_key,
            environment=self.environment,
            cloud=self.cloud,
            region=self.region,
            metric=self.metric,
            namespace=self.namespace
        )
    
    def _create_enhanced_manager(self):
        """Create enhanced Pinecone manager instance."""
        from .pinecone_enhanced import PineconeEnhancedManager
        return PineconeEnhancedManager(
            index_name=self.index_name,
            api_key=self.api_key,
            environment=self.environment,
            cloud=self.cloud,
            region=self.region,
            metric=self.metric,
            namespace=self.namespace
        )
    
    # --- Delegate all methods to the underlying implementation ---
    
    def ensure_index(self, dimension: int) -> None:
        """Ensure index exists with given dimension."""
        return self._impl.ensure_index(dimension)
    
    def get_index_dimension(self) -> Optional[int]:
        """Get the dimension of the index."""
        return self._impl.get_index_dimension()
    
    def upsert_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Upload vectors to the index with optional quality gates."""
        return self._impl.upsert_vectors(ids, vectors, metadatas, namespace, **kwargs)
    
    def upsert_items(
        self,
        items: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Upload items to the index with optional quality gates."""
        return self._impl.upsert_items(items, namespace, **kwargs)
    
    def query_by_vector(
        self,
        vector: List[float],
        top_k: int = 5,
        include_metadata: bool = True,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query for similar vectors."""
        return self._impl.query_by_vector(vector, top_k, include_metadata, namespace, filter)
    
    def query_by_text(
        self,
        text: str,
        embedder: "Embedder",
        top_k: int = 5,
        include_metadata: bool = True,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Query for similar texts via embedding with optional reranking."""
        return self._impl.query_by_text(text, embedder, top_k, include_metadata, namespace, filter, **kwargs)
    
    def delete_by_document_id(self, doc_id: str, namespace: Optional[str] = None) -> Any:
        """Delete document by document ID."""
        return self._impl.delete_by_document_id(doc_id, namespace)
    
    # --- Legacy compatibility methods ---
    
    def upsert_document(self, **kwargs) -> Any:
        """Legacy wrapper for upsert functionality."""
        return self._impl.upsert_document(**kwargs)
    
    def search_similar_documents(
        self,
        query_vector: List[float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Legacy wrapper for search functionality."""
        return self._impl.search_similar_documents(query_vector, top_k, namespace, filter_dict)
    
    # --- Enhanced methods (if available) ---
    
    def batch_upsert_with_retry(self, items: List[Dict[str, Any]], **kwargs) -> Any:
        """Batch upsert with retry logic (enhanced feature)."""
        if hasattr(self._impl, 'batch_upsert_with_retry'):
            return self._impl.batch_upsert_with_retry(items, **kwargs)
        else:
            # Fallback to simple upsert
            return self._impl.upsert_items(items, **kwargs)
    
    def monitor_performance(self) -> Dict[str, Any]:
        """Monitor performance metrics (enhanced feature)."""
        if hasattr(self._impl, 'monitor_performance'):
            return self._impl.monitor_performance()
        else:
            return {"implementation": "simple", "enhanced_monitoring": False}
    
    def optimize_index(self) -> bool:
        """Optimize index performance (enhanced feature)."""
        if hasattr(self._impl, 'optimize_index'):
            return self._impl.optimize_index()
        else:
            return False
    
    # --- Utility properties ---
    
    @property
    def implementation_type(self) -> str:
        """Get the type of implementation being used."""
        if hasattr(self._impl, '__class__'):
            return self._impl.__class__.__name__
        return "unknown"
    
    @property
    def is_enhanced(self) -> bool:
        """Check if enhanced implementation is being used."""
        return "Enhanced" in self.implementation_type


# Factory function for compatibility
def get_pinecone_manager(**kwargs) -> PineconeManager:
    """Factory function to create PineconeManager facade."""
    return PineconeManager(**kwargs)


def make_pinecone_manager(
    index_name: str,
    api_key: Optional[str] = None,
    environment: Optional[str] = None,  # v2
    cloud: Optional[str] = None,        # v3 serverless
    region: Optional[str] = None,       # v3 serverless
    metric: str = "cosine",
    namespace: Optional[str] = None,
    force_simple: bool = False
) -> PineconeManager:
    """
    Standardized factory function for creating Pinecone managers.
    
    This is the recommended way to create Pinecone managers across
    all CLI/Worker/API components to ensure consistent wiring.
    
    Args:
        index_name: Name of the Pinecone index
        api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
        environment: Pinecone environment for v2 (defaults to PINECONE_ENV env var)
        cloud: Cloud provider for v3 serverless (defaults to PINECONE_CLOUD env var)
        region: Region for v3 serverless (defaults to PINECONE_REGION env var)
        metric: Distance metric (default: cosine)
        namespace: Default namespace (defaults to PINECONE_NAMESPACE env var)
        force_simple: Force simple implementation (default: False)
    
    Returns:
        PineconeManager facade instance
        
    Example:
        from bu_processor.integrations.pinecone_facade import make_pinecone_manager
        from bu_processor.embeddings.embedder import Embedder
        
        embedder = Embedder()
        pc = make_pinecone_manager(
            index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV"),  # v2
            cloud=os.getenv("PINECONE_CLOUD"),      # v3
            region=os.getenv("PINECONE_REGION"),    # v3
            namespace=os.getenv("PINECONE_NAMESPACE")
        )
        pc.ensure_index(embedder.dimension)
    """
    import os
    
    # Use environment variables as defaults
    api_key = api_key or os.getenv("PINECONE_API_KEY")
    environment = environment or os.getenv("PINECONE_ENV")
    cloud = cloud or os.getenv("PINECONE_CLOUD")
    region = region or os.getenv("PINECONE_REGION")
    namespace = namespace or os.getenv("PINECONE_NAMESPACE")
    
    return PineconeManager(
        index_name=index_name,
        api_key=api_key,
        environment=environment,
        cloud=cloud,
        region=region,
        metric=metric,
        namespace=namespace,
        force_simple=force_simple
    )


# Exports
__all__ = [
    "PineconeManager",
    "get_pinecone_manager",
    "make_pinecone_manager"
]
