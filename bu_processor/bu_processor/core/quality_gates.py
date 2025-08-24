# bu_processor/core/quality_gates.py
"""
Quality Gates and Consistency Checks
=====================================

This module provides quality gates and consistency checks for the ML classifier system.
These serve as safeguards to prevent "dumb" simplifications and ensure data integrity.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from ..embeddings.embedder import Embedder

logger = logging.getLogger(__name__)

class QualityGateError(Exception):
    """Raised when a quality gate check fails."""
    pass

def check_dimension_consistency(
    pinecone_manager: Any,
    embedder: "Embedder"
) -> None:
    """
    Quality Gate: Check dimension consistency before first upsert.
    
    This prevents dimension mismatches that could cause runtime errors
    or silent data corruption.
    
    Args:
        pinecone_manager: PineconeManager instance (simple or enhanced)
        embedder: Embedder instance
        
    Raises:
        QualityGateError: If dimensions don't match
        RuntimeError: If index dimension cannot be determined
    """
    try:
        # Get dimensions from both sources
        idx_dim = pinecone_manager.get_index_dimension()
        emb_dim = embedder.dimension
        
        logger.debug(
            "Dimension consistency check",
            index_dimension=idx_dim,
            embedding_dimension=emb_dim
        )
        
        # Quality gate: dimensions must match if index exists
        if idx_dim is not None and idx_dim != emb_dim:
            raise QualityGateError(
                f"Dimension mismatch: Index expects {idx_dim} dimensions, "
                f"but embedder produces {emb_dim} dimensions. "
                f"This would cause data corruption or runtime errors."
            )
        
        # If index doesn't exist yet, ensure it gets created with correct dimension
        if idx_dim is None:
            logger.info(
                "Index not found, will be created with embedding dimension",
                dimension=emb_dim
            )
            pinecone_manager.ensure_index(dimension=emb_dim)
        
        logger.info("✅ Dimension consistency check passed")
        
    except Exception as e:
        if isinstance(e, QualityGateError):
            raise
        # Wrap other exceptions for clarity
        raise RuntimeError(f"Failed to check dimension consistency: {e}") from e

def validate_upsert_data(
    ids: Optional[List[str]] = None,
    vectors: Optional[List[List[float]]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    items: Optional[List[Dict[str, Any]]] = None
) -> None:
    """
    Quality Gate: Validate upsert data before processing.
    
    Args:
        ids: Vector IDs
        vectors: Vector embeddings
        metadatas: Metadata dictionaries
        items: Item format data
        
    Raises:
        QualityGateError: If data validation fails
    """
    if items is not None:
        # Validate items format
        if not isinstance(items, list) or len(items) == 0:
            raise QualityGateError("Items must be a non-empty list")
        
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise QualityGateError(f"Item {i} must be a dictionary")
            
            if "id" not in item:
                raise QualityGateError(f"Item {i} missing required 'id' field")
            
            if "values" not in item:
                raise QualityGateError(f"Item {i} missing required 'values' field")
            
            if not isinstance(item["values"], list):
                raise QualityGateError(f"Item {i} 'values' must be a list")
    
    else:
        # Validate vectors format
        if not ids or not vectors:
            raise QualityGateError("Both 'ids' and 'vectors' are required for vector format")
        
        if len(ids) != len(vectors):
            raise QualityGateError(
                f"Length mismatch: {len(ids)} ids vs {len(vectors)} vectors"
            )
        
        if metadatas and len(metadatas) != len(ids):
            raise QualityGateError(
                f"Length mismatch: {len(ids)} ids vs {len(metadatas)} metadatas"
            )
        
        # Check vector dimensions are consistent
        if vectors:
            first_dim = len(vectors[0]) if vectors[0] else 0
            for i, vector in enumerate(vectors):
                if len(vector) != first_dim:
                    raise QualityGateError(
                        f"Vector {i} has {len(vector)} dimensions, "
                        f"expected {first_dim} (inconsistent vector dimensions)"
                    )
    
    logger.debug("✅ Upsert data validation passed")

def apply_quality_gates(
    pinecone_manager: Any,
    embedder: "Embedder",
    ids: Optional[List[str]] = None,
    vectors: Optional[List[List[float]]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    items: Optional[List[Dict[str, Any]]] = None,
    skip_dimension_check: bool = False
) -> None:
    """
    Apply all quality gates before upsert operation.
    
    This function serves as a single entry point for all quality checks,
    preventing "dumb" simplifications and ensuring data integrity.
    
    Args:
        pinecone_manager: PineconeManager instance
        embedder: Embedder instance
        ids: Vector IDs (for vector format)
        vectors: Vector embeddings (for vector format)
        metadatas: Metadata dictionaries (for vector format)
        items: Item format data
        skip_dimension_check: Skip dimension consistency check (for testing)
        
    Raises:
        QualityGateError: If any quality gate fails
        RuntimeError: If quality gates cannot be executed
    """
    logger.debug("Applying quality gates before upsert")
    
    # Quality Gate 1: Dimension consistency check
    if not skip_dimension_check:
        check_dimension_consistency(pinecone_manager, embedder)
    
    # Quality Gate 2: Data validation
    validate_upsert_data(ids, vectors, metadatas, items)
    
    logger.info("✅ All quality gates passed")

# Exports
__all__ = [
    "QualityGateError",
    "check_dimension_consistency", 
    "validate_upsert_data",
    "apply_quality_gates"
]
