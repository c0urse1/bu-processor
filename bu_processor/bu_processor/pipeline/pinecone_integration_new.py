#!/usr/bin/env python3
"""
🌲 SIMPLIFIED PINECONE INTEGRATION (Legacy Compatibility)
========================================================

This file now provides backward compatibility for existing code.
The main implementation has been moved to bu_processor.integrations.pinecone_manager.

REMOVED FOR MVP:
- Prometheus metrics  
- Rate limiting
- Embedding cache
- Stub mode complexity
- Multiple PineconeManager classes
- Async complexity
"""

from __future__ import annotations

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from ..core.logging_setup import get_logger

logger = get_logger("pinecone_integration")

# Import the simplified manager
from ..integrations.pinecone_manager import (
    PineconeManager as SimplifiedPineconeManager,
    get_pinecone_manager
)

# =============================================================================
# ENVIRONMENT VARIABLES FOR COMPATIBILITY
# =============================================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX", "bu-processor-embeddings")

# SDK availability check for compatibility
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_SDK_AVAILABLE = True
except ImportError:
    PINECONE_SDK_AVAILABLE = False

PINECONE_AVAILABLE = bool(PINECONE_API_KEY) and PINECONE_SDK_AVAILABLE
PINECONE_ASYNC_AVAILABLE = False  # Simplified for MVP

# =============================================================================
# DATA CLASSES FOR BACKWARD COMPATIBILITY
# =============================================================================

@dataclass
class VectorSearchResult:
    """Result from a vector similarity search - backward compatibility."""
    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def doc_id(self) -> str:
        """Alias for id to maintain compatibility."""
        return self.id

@dataclass
class DocumentEmbedding:
    """Document with its embedding vector."""
    id: str
    text: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# MAIN COMPATIBILITY CLASSES
# =============================================================================

class PineconeManager:
    """
    Legacy compatibility wrapper around the simplified PineconeManager.
    This maintains the old interface while delegating to the new implementation.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with backward compatibility for old signatures."""
        logger.info("Using simplified PineconeManager (legacy compatibility mode)")
        
        # Handle various legacy parameter patterns
        api_key = kwargs.get('api_key') or (args[0] if args else None)
        index_name = kwargs.get('index_name', DEFAULT_INDEX_NAME)
        
        # Extract relevant parameters and ignore legacy ones like stub_mode, prometheus, etc.
        clean_kwargs = {
            'index_name': index_name,
            'api_key': api_key,
            'environment': kwargs.get('environment'),
            'cloud': kwargs.get('cloud'),
            'region': kwargs.get('region'),
            'metric': kwargs.get('metric', 'cosine'),
            'namespace': kwargs.get('namespace')
        }
        
        # Remove None values
        clean_kwargs = {k: v for k, v in clean_kwargs.items() if v is not None}
        
        self._manager = SimplifiedPineconeManager(**clean_kwargs)
    
    def ensure_index(self, dimension: int) -> None:
        """Ensure index exists with given dimension."""
        return self._manager.ensure_index(dimension)
    
    def upsert_vectors(self, vectors: List[tuple]) -> Dict[str, Any]:
        """Upsert vectors with legacy compatibility."""
        # Handle both new signature (ids, vectors, metadatas) and old tuple format
        if isinstance(vectors[0], tuple):
            # Old format: [(id, vector, metadata), ...]
            ids = [v[0] for v in vectors]
            vecs = [v[1] for v in vectors]
            metas = [v[2] if len(v) > 2 else {} for v in vectors]
            return self._manager.upsert_vectors(ids, vecs, metas)
        else:
            # Assume it's already in the new format
            return self._manager.upsert_vectors(vectors)
    
    def search_similar_documents(self, query_vector: List[float], top_k: int = 10, **kwargs) -> List[VectorSearchResult]:
        """Search with legacy compatibility."""
        results = self._manager.search_similar_documents(query_vector, top_k, **kwargs)
        
        # Convert to legacy format
        return [
            VectorSearchResult(
                id=result["doc_id"],
                score=result["score"],
                metadata=result["metadata"]
            )
            for result in results
        ]
    
    def delete_vectors(self, vector_ids: List[str], **kwargs) -> Dict[str, Any]:
        """Delete vectors with legacy compatibility."""
        # For now, use delete_by_document_id for each ID
        results = []
        for doc_id in vector_ids:
            result = self._manager.delete_by_document_id(doc_id, **kwargs)
            results.append(result)
        return {"success": True, "deleted_count": len(vector_ids)}
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index stats."""
        try:
            return {"dimension": self._manager.get_index_dimension()}
        except:
            return {}

# =============================================================================
# LEGACY ALIASES AND STUBS
# =============================================================================

# For tests that expect these classes
AsyncPineconeManager = PineconeManager  # Legacy alias
LegacyPineconeManager = PineconeManager  # Legacy alias
PineconeManagerStub = PineconeManager  # Simplified - no separate stub needed

# Legacy config classes - simplified stubs
class AsyncPineconeConfig:
    """Legacy config stub."""
    def __init__(self, **kwargs):
        pass

class AsyncPineconePipeline:
    """Legacy pipeline stub."""
    def __init__(self, **kwargs):
        pass

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def DefaultPineconeManager(**kwargs) -> PineconeManager:
    """Factory function for default manager."""
    return PineconeManager(**kwargs)

def PineconeManagerAlias(**kwargs) -> PineconeManager:
    """Alias factory function."""
    return PineconeManager(**kwargs)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core manager classes
    "AsyncPineconeManager",
    "PineconeManager",
    "PineconeManagerStub", 
    "LegacyPineconeManager",
    
    # Factory functions
    "get_pinecone_manager",
    "DefaultPineconeManager",
    "PineconeManagerAlias",
    
    # Data classes
    "VectorSearchResult",
    "DocumentEmbedding",
    
    # Legacy stubs
    "AsyncPineconeConfig",
    "AsyncPineconePipeline",
    
    # Environment flags
    "PINECONE_AVAILABLE",
    "PINECONE_SDK_AVAILABLE",
    "PINECONE_ASYNC_AVAILABLE",
    "DEFAULT_INDEX_NAME"
]
