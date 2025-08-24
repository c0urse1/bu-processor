# bu_processor/core/reranking.py
"""
Reranking Module - Intelligence Booster
========================================

This module provides optional reranking functionality using cross-encoders
to improve search quality. Controlled by ENABLE_RERANK flag.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import logging
from .flags import ENABLE_RERANK

logger = logging.getLogger(__name__)

class RerankerError(Exception):
    """Raised when reranking operations fail."""
    pass

class CrossEncoderReranker:
    """
    Cross-encoder based reranker for improving search results.
    
    Uses a cross-encoder model to compute relevance scores between
    queries and retrieved documents, then reorders results accordingly.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self._model = None
        
        if ENABLE_RERANK:
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info(f"✅ Cross-encoder reranker initialized with {self.model_name}")
        except ImportError as e:
            logger.warning(
                "Cross-encoder reranking requires sentence-transformers",
                error=str(e)
            )
            raise RerankerError(
                "Cross-encoder reranking requires sentence-transformers package"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder model: {e}")
            raise RerankerError(f"Failed to initialize reranker: {e}") from e
    
    def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        text_field: str = "text",
        score_field: str = "score",
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using cross-encoder scoring.
        
        Args:
            query: Original search query
            results: List of search results from Pinecone
            text_field: Field name containing the text content
            score_field: Field name containing the original score
            max_results: Maximum number of results to return
            
        Returns:
            Reranked results with updated scores
        """
        if not ENABLE_RERANK:
            logger.debug("Reranking disabled, returning original results")
            return results[:max_results] if max_results else results
        
        if not self._model:
            logger.warning("Reranker not initialized, returning original results")
            return results[:max_results] if max_results else results
        
        if not results:
            return results
        
        try:
            # Extract texts for reranking
            texts = []
            for result in results:
                # Try to get text from metadata or direct field
                text = None
                if isinstance(result.get("metadata"), dict):
                    text = result["metadata"].get(text_field)
                if not text:
                    text = result.get(text_field, "")
                
                if not text:
                    logger.warning(f"No text found for result with ID: {result.get('id', 'unknown')}")
                    text = ""
                
                texts.append(text)
            
            # Create query-document pairs for cross-encoder
            query_doc_pairs = [(query, text) for text in texts]
            
            logger.debug(f"Reranking {len(query_doc_pairs)} results with cross-encoder")
            
            # Get cross-encoder scores
            ce_scores = self._model.predict(query_doc_pairs)
            
            # Update results with cross-encoder scores
            reranked_results = []
            for i, result in enumerate(results):
                enhanced_result = result.copy()
                enhanced_result["original_score"] = result.get(score_field, 0.0)
                enhanced_result["cross_encoder_score"] = float(ce_scores[i])
                enhanced_result[score_field] = float(ce_scores[i])  # Use CE score as primary
                reranked_results.append(enhanced_result)
            
            # Sort by cross-encoder score (descending)
            reranked_results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
            
            logger.info(
                f"✅ Reranked {len(results)} results using cross-encoder",
                model=self.model_name
            )
            
            # Return top results if max_results specified
            return reranked_results[:max_results] if max_results else reranked_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to original results on error
            return results[:max_results] if max_results else results

def rerank_search_results(
    query: str,
    results: List[Dict[str, Any]],
    reranker: Optional[CrossEncoderReranker] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function to rerank search results.
    
    Args:
        query: Original search query
        results: List of search results
        reranker: Optional reranker instance (creates default if None)
        **kwargs: Additional arguments for reranking
        
    Returns:
        Reranked results
    """
    if not ENABLE_RERANK:
        return results
    
    if not reranker:
        try:
            reranker = CrossEncoderReranker()
        except RerankerError:
            logger.warning("Failed to create reranker, returning original results")
            return results
    
    return reranker.rerank_results(query, results, **kwargs)

# Exports
__all__ = [
    "RerankerError",
    "CrossEncoderReranker", 
    "rerank_search_results"
]
