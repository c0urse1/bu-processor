#!/usr/bin/env python3
"""
🌳 SEMANTIC CLUSTERING ENHANCEMENT - ROBUST & API-STABLE
========================================================

Enhanced semantic clustering for hierarchical chunks with unified API.
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# =============================================================================
# DEPENDENCY HANDLING WITH GRACEFUL FALLBACKS
# =============================================================================

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    SentenceTransformer = None
    _HAS_SBERT = False

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _HAS_SKLEARN = True
except ImportError:
    KMeans = DBSCAN = TfidfVectorizer = cosine_similarity = np = None
    _HAS_SKLEARN = False

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SemanticClusterResult:
    """Result of semantic clustering operation"""
    cluster_assignments: List[int]
    cluster_centers: Optional[List[List[float]]] = None
    silhouette_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class ClusteringMethod(Enum):
    """Available clustering methods"""
    SEMANTIC_KMEANS = "semantic_kmeans"
    TFIDF_CLUSTERING = "tfidf_clustering"
    FALLBACK_SIMPLE = "fallback_simple"

# =============================================================================
# MAIN CLASS - SEMANTIC CLUSTERING ENHANCER
# =============================================================================

class SemanticClusteringEnhancer:
    """Enhanced semantic clustering with unified API and robust fallbacks"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        clustering_method: str = "auto"
    ):
        """
        Initialize the semantic clustering enhancer
        
        Args:
            model_name: SentenceTransformer model name (auto-select if None)
            clustering_method: Preferred clustering method or "auto"
        """
        self.logger = logger
        self.model_name = model_name or "distiluse-base-multilingual-cased"
        self.embedding_model = None
        
        # Initialize embedding model if available
        if _HAS_SBERT:
            try:
                self.embedding_model = SentenceTransformer(self.model_name)
                self.logger.info(f"Loaded SentenceTransformer: {self.model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load SentenceTransformer: {e}")
        
        # Determine clustering method
        self.clustering_method = self._init_clustering_method(clustering_method)
        
        self.logger.info(f"SemanticClusteringEnhancer initialized with method: {self.clustering_method.value}")

    def _init_clustering_method(self, method: str) -> ClusteringMethod:
        """Initialize clustering method based on available dependencies"""
        if method == "auto":
            if _HAS_SBERT and _HAS_SKLEARN:
                return ClusteringMethod.SEMANTIC_KMEANS
            elif _HAS_SKLEARN:
                return ClusteringMethod.TFIDF_CLUSTERING
            else:
                return ClusteringMethod.FALLBACK_SIMPLE
        
        # Try to use requested method
        try:
            requested = ClusteringMethod(method)
            if self._method_available(requested):
                return requested
            else:
                self.logger.warning(f"Method {method} not available, using auto-selection")
                return self._init_clustering_method("auto")
        except ValueError:
            self.logger.warning(f"Unknown method {method}, using auto-selection")
            return self._init_clustering_method("auto")

    def _method_available(self, method: ClusteringMethod) -> bool:
        """Check if clustering method is available"""
        if method == ClusteringMethod.SEMANTIC_KMEANS:
            return _HAS_SBERT and _HAS_SKLEARN
        elif method == ClusteringMethod.TFIDF_CLUSTERING:
            return _HAS_SKLEARN
        elif method == ClusteringMethod.FALLBACK_SIMPLE:
            return True
        return False

    # =========================================================================
    # UNIFIED API METHODS
    # =========================================================================

    def cluster_texts(
        self,
        texts: List[str],
        num_clusters: Optional[int] = None,
        # Legacy compatibility
        n_clusters: Optional[int] = None
    ) -> SemanticClusterResult:
        """
        UNIFIED API: Cluster texts using the best available method
        
        Args:
            texts: List of text strings to cluster
            num_clusters: Number of clusters (auto-determined if None)
            n_clusters: Legacy parameter name (for backward compatibility)
            
        Returns:
            SemanticClusterResult with cluster assignments and metadata
        """
        # Handle legacy parameter
        if n_clusters is not None and num_clusters is None:
            num_clusters = n_clusters
        
        # Auto-determine cluster count if not specified
        if num_clusters is None:
            num_clusters = max(2, min(10, len(texts) // 5))
        
        start_time = time.time()
        
        try:
            if self.clustering_method == ClusteringMethod.SEMANTIC_KMEANS:
                result = self._cluster_semantic_kmeans(texts, num_clusters)
            elif self.clustering_method == ClusteringMethod.TFIDF_CLUSTERING:
                result = self._cluster_tfidf(texts, num_clusters)
            else:
                result = self._cluster_fallback(texts, num_clusters)
            
            # Add timing metadata
            execution_time = time.time() - start_time
            if result.metadata is None:
                result.metadata = {}
            result.metadata['execution_time'] = execution_time
            result.metadata['method_used'] = self.clustering_method.value
            
            return result
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            # Return fallback result
            return SemanticClusterResult(
                cluster_assignments=[0] * len(texts),
                metadata={'error': str(e), 'fallback_used': True}
            )

    def calculate_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        UNIFIED API: Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if self.embedding_model and _HAS_SKLEARN:
                # Use SBERT embeddings with cosine similarity
                embeddings = self.embedding_model.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(similarity)
            elif _HAS_SKLEARN:
                # Use TF-IDF similarity
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return float(similarity)
            else:
                # Simple token-based similarity
                tokens1 = set(text1.lower().split())
                tokens2 = set(text2.lower().split())
                if not tokens1 and not tokens2:
                    return 1.0
                if not tokens1 or not tokens2:
                    return 0.0
                intersection = len(tokens1 & tokens2)
                union = len(tokens1 | tokens2)
                return intersection / union
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    # =========================================================================
    # AVAILABILITY HELPER FUNCTIONS
    # =========================================================================

    def is_semantic_available(self) -> bool:
        """Check if semantic clustering capabilities are available"""
        return _HAS_SBERT and self.embedding_model is not None

    def get_available_methods(self) -> List[str]:
        """Get list of available clustering methods"""
        methods = []
        if _HAS_SBERT and _HAS_SKLEARN:
            methods.append("semantic_kmeans")
        if _HAS_SKLEARN:
            methods.append("tfidf_clustering")
        methods.append("fallback_simple")
        return methods

    def get_capabilities(self) -> Dict[str, Any]:
        """Get detailed information about available capabilities"""
        return {
            'has_sentence_transformers': _HAS_SBERT,
            'has_sklearn': _HAS_SKLEARN,
            'embedding_model_loaded': self.embedding_model is not None,
            'current_method': self.clustering_method.value,
            'available_methods': self.get_available_methods(),
            'model_name': self.model_name
        }

    # =========================================================================
    # INTERNAL CLUSTERING IMPLEMENTATIONS
    # =========================================================================

    def _cluster_semantic_kmeans(self, texts: List[str], num_clusters: int) -> SemanticClusterResult:
        """Cluster using SentenceTransformers + KMeans"""
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_assignments = kmeans.fit_predict(embeddings).tolist()
            
            # Calculate silhouette score if more than 1 cluster
            silhouette_score = None
            if num_clusters > 1 and len(set(cluster_assignments)) > 1:
                try:
                    from sklearn.metrics import silhouette_score as sk_silhouette_score
                    silhouette_score = sk_silhouette_score(embeddings, cluster_assignments)
                except ImportError:
                    pass
            
            return SemanticClusterResult(
                cluster_assignments=cluster_assignments,
                cluster_centers=kmeans.cluster_centers_.tolist(),
                silhouette_score=silhouette_score
            )
        except Exception as e:
            self.logger.error(f"Semantic KMeans clustering failed: {e}")
            raise

    def _cluster_tfidf(self, texts: List[str], num_clusters: int) -> SemanticClusterResult:
        """Cluster using TF-IDF + KMeans"""
        try:
            # Generate TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_assignments = kmeans.fit_predict(tfidf_matrix).tolist()
            
            return SemanticClusterResult(
                cluster_assignments=cluster_assignments,
                cluster_centers=kmeans.cluster_centers_.tolist()
            )
        except Exception as e:
            self.logger.error(f"TF-IDF clustering failed: {e}")
            raise

    def _cluster_fallback(self, texts: List[str], num_clusters: int) -> SemanticClusterResult:
        """Simple fallback clustering using text length and basic similarity"""
        try:
            # Simple clustering based on text length and basic token similarity
            text_lengths = [len(text) for text in texts]
            
            # Create clusters based on text length quartiles
            if num_clusters <= 1:
                cluster_assignments = [0] * len(texts)
            else:
                # Sort by length and assign to clusters
                sorted_indices = sorted(range(len(texts)), key=lambda i: text_lengths[i])
                cluster_size = len(texts) / num_clusters
                
                cluster_assignments = [0] * len(texts)
                for i, idx in enumerate(sorted_indices):
                    cluster_id = min(int(i / cluster_size), num_clusters - 1)
                    cluster_assignments[idx] = cluster_id
            
            return SemanticClusterResult(cluster_assignments=cluster_assignments)
        except Exception as e:
            self.logger.error(f"Fallback clustering failed: {e}")
            # Ultimate fallback - all texts in one cluster
            return SemanticClusterResult(cluster_assignments=[0] * len(texts))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def is_semantic_available() -> bool:
    """Check if semantic clustering capabilities are available globally"""
    return _HAS_SBERT and _HAS_SKLEARN

def get_semantic_capabilities() -> Dict[str, bool]:
    """Get information about available semantic processing capabilities"""
    return {
        'sentence_transformers': _HAS_SBERT,
        'sklearn': _HAS_SKLEARN,
        'semantic_clustering': _HAS_SBERT and _HAS_SKLEARN,
        'tfidf_clustering': _HAS_SKLEARN,
        'basic_clustering': True
    }


if __name__ == "__main__":
    # Quick test
    enhancer = SemanticClusteringEnhancer()
    print(f"Capabilities: {enhancer.get_capabilities()}")
    
    # Test with sample texts
    test_texts = [
        "This is about machine learning and AI",
        "The weather is nice today",
        "Deep learning models are powerful",
        "I love sunny days",
        "Neural networks are fascinating"
    ]
    
    result = enhancer.cluster_texts(test_texts, num_clusters=2)
    print(f"Cluster assignments: {result.cluster_assignments}")
    
    # Test similarity
    sim = enhancer.calculate_similarity(test_texts[0], test_texts[2])
    print(f"Similarity between ML texts: {sim:.3f}")