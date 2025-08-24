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
    from ..core.flags import ENABLE_ENHANCED_PINECONE
    from .pinecone_enhanced import PineconeEnhancedManager
    from .pinecone_simple import PineconeManager as SimplePineconeManager
except ImportError:
    # Fallback if flags not available
    ENABLE_ENHANCED_PINECONE = False
    from .pinecone_simple import PineconeManager as SimplePineconeManager

class PineconeManager:
    """
    Robust facade for Pinecone operations that automatically selects
    the appropriate implementation (simple or enhanced) based on feature flags.
    Falls back to simple implementation if enhanced fails.
    """
    
    def __init__(self, *args, **kwargs):
        # Facade-only flags abfangen (nicht weiterreichen!)
        force_simple   = kwargs.pop("force_simple", None)
        force_enhanced = kwargs.pop("force_enhanced", None)

        if force_simple and force_enhanced:
            raise ValueError("force_simple und force_enhanced schließen sich gegenseitig aus.")

        # Explizite Steuerung hat Vorrang
        if force_simple is True:
            self._impl = SimplePineconeManager(*args, **kwargs)
            return
        if force_enhanced is True:
            try:
                self._impl = PineconeEnhancedManager(*args, **kwargs)
                return
            except Exception:
                # Fallback auf Simple, wenn Enhanced fehlschlägt
                self._impl = SimplePineconeManager(*args, **kwargs)
                return

        # Sonst per Feature-Flag wählen
        if ENABLE_ENHANCED_PINECONE:
            try:
                self._impl = PineconeEnhancedManager(*args, **kwargs)
                return
            except Exception:
                # Safety net: fällt auf Simple zurück
                self._impl = SimplePineconeManager(*args, **kwargs)
                return

        # Default: Simple
        self._impl = SimplePineconeManager(*args, **kwargs)

    # Delegation
    def ensure_index(self, *a, **kw):         
        return self._impl.ensure_index(*a, **kw)
    
    def get_index_dimension(self, *a, **kw):  
        return self._impl.get_index_dimension(*a, **kw)
    
    def upsert_vectors(self, *a, **kw):       
        return self._impl.upsert_vectors(*a, **kw)
    
    def upsert_items(self, *a, **kw):         
        return self._impl.upsert_items(*a, **kw)
    
    def query_by_vector(self, *a, **kw):      
        return self._impl.query_by_vector(*a, **kw)
    
    def query_by_text(self, *a, **kw):        
        return self._impl.query_by_text(*a, **kw)
    
    def delete_by_document_id(self, *a, **kw):
        return self._impl.delete_by_document_id(*a, **kw)


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
    **kwargs  # Allow force_* flags to be passed through if caller wants them
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
        **kwargs: Additional arguments including optional force_simple/force_enhanced flags
    
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
    
    # KEINE force_* Flags hier einfügen – nur durchreichen, falls der Aufrufer sie bewusst übergibt
    return PineconeManager(
        index_name=index_name,
        api_key=api_key,
        environment=environment,
        cloud=cloud,
        region=region,
        metric=metric,
        namespace=namespace,
        **kwargs  # Pass through any additional kwargs including force flags
    )


# Exports
__all__ = [
    "PineconeManager",
    "get_pinecone_manager",
    "make_pinecone_manager"
]
