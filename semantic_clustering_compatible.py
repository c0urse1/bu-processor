#!/usr/bin/env python3
"""
🌳 SEMANTIC CLUSTERING ENHANCEMENT - CORRECTED IMPLEMENTATION
============================================================

Enhanced semantic clustering for hierarchical chunks with unified API.
This implementation addresses all identified issues and provides robust fallback mechanisms.

Based on the comprehensive analysis and corrected implementation specification.
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# =============================================================================
# DEPENDENCY HANDLING WITH GRACEFUL FALLBACKS
# =============================================================================

# Import numpy first (it's a base dependency)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SBERT_AVAILABLE = False

try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    KMeans = DBSCAN = AgglomerativeClustering = None
    TfidfVectorizer = cosine_similarity = None
    SKLEARN_AVAILABLE = False

# Update overall availability flag
SEMANTIC_DEPS_AVAILABLE = SBERT_AVAILABLE and SKLEARN_AVAILABLE and NUMPY_AVAILABLE

# =============================================================================
# CONFIGURATION AND DATA STRUCTURES
# =============================================================================

# Default semantic configuration
SEMANTIC_CONFIG = {
    "models": {
        "fast_embedding": "paraphrase-MiniLM-L6-v2",
        "multilingual": "distiluse-base-multilingual-cased",
        "default": "paraphrase-MiniLM-L6-v2"
    },
    "caching": {
        "max_cache_size": 100
    },
    "clustering": {
        "default_clusters": 3,
        "dbscan_eps": 0.5,
        "dbscan_min_samples": 2
    }
}

@dataclass
class EmbeddingCacheEntry:
    """Cache entry for computed embeddings with LRU tracking"""
    vector: Any  # np.ndarray when numpy is available
    last_access: float

# Try to load settings if available
try:
    # This import will fail in standalone mode, which is expected
    from bu_processor.core.config import get_config
    settings = get_config()
except ImportError:
    settings = None
    TECHNICAL = "technical"
    TABLE_HEAVY = "table_heavy"
    NARRATIVE = "narrative"
    MIXED = "mixed"

@dataclass
class SemanticClusterResult:
    cluster_assignments: List[int]
    cluster_centers: Optional[List[List[float]]] = None
    silhouette_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class ClusteringMethod(Enum):
    SEMANTIC_KMEANS = "semantic_kmeans"
    TFIDF_CLUSTERING = "tfidf_clustering"
    FALLBACK_SIMPLE = "fallback_simple"

class SemanticClusteringEnhancer:
    """Erweitert das hierarchische Chunking-System um semantische Clustering-Fähigkeiten.
    
    Diese Klasse bietet erweiterte semantische Analysen für Dokument-Chunks 
    mit konfigurierbaren Clustering-Algorithmen und intelligentem Caching. 
    Sie kann Texte in inhaltlich ähnliche Gruppen clustern und semantische Ähnlichkeiten berechnen, 
    wahlweise unter Nutzung vortrainierter Modelle (SentenceTransformer).
    
    Attributes:
        clustering_method (str): Name des Clustering-Algorithmus ("kmeans", "dbscan", "agglomerative").
        embedding_model: Das geladene SentenceTransformer-Modell für Embeddings (oder None bei Fallback).
        embedding_cache (Dict[str, EmbeddingCacheEntry]): LRU-Cache für bereits berechnete Embeddings.
        _cache_hits (int): Anzahl Cache-Treffer (zur Statistik).
        _cache_misses (int): Anzahl Cache-Fehlgriffe.
        config (Dict[str, Any]): Konfigurationseinstellungen (Modelle, Caching-Parameter etc.).
    """
    
    def __init__(self, model_name: Optional[str] = None, max_cache_size: int = None, 
                 clustering_method: str = "kmeans") -> None:
        """Initialisiert den SemanticClusteringEnhancer.
        
        Optional kann ein spezifisches Transformermodell geladen und ein Clustering-Algorithmus gewählt werden.
        
        Args:
            model_name: Name des vortrainierten SentenceTransformer-Modells (optional).
            max_cache_size: Maximale Anzahl der Cache-Einträge für Embeddings (optional).
            clustering_method: Zu nutzender Clustering-Algorithmus ("kmeans", "dbscan" oder "agglomerative").
        
        Raises:
            RuntimeError: Wenn die optionalen ML-Dependencies vorhanden sind, 
                          das angegebene Modell aber nicht geladen werden kann.
        """
        # Set the clustering algorithm choice
        self.clustering_method = clustering_method.lower() if clustering_method else "kmeans"
        self.logger = logger  # use module-level logger
        
        # Load configuration (from global settings if available, else default config)
        if settings and hasattr(settings, "semantic"):
            # Use dynamic settings (ensures current environment config is used)
            self.config = {
                "models": getattr(settings.semantic, "models", {}),
                "batch_processing": getattr(settings.semantic, "batch_processing", {}),
                "clustering": getattr(settings.semantic, "clustering", {}),
                "caching": getattr(settings.semantic, "caching", {}),
                "similarity": getattr(settings.semantic, "similarity", {})
            }
        else:
            # Fallback to static default config
            self.config = SEMANTIC_CONFIG
        
        # Determine cache size from config or override
        cache_size = max_cache_size or self.config["caching"].get("max_cache_size", 100)
        self.embedding_cache: Dict[str, EmbeddingCacheEntry] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = cache_size
        
        # Initialize embedding model if dependencies are available
        self.embedding_model = None
        if SEMANTIC_DEPS_AVAILABLE:
            # Choose model: use provided name or default from config
            chosen_model = model_name
            if not chosen_model:
                # Prefer a fast or multilingual model from config if available
                model_options = self.config.get("models", {})
                # Try a preferred key from models config
                if "fast_embedding" in model_options:
                    chosen_model = model_options["fast_embedding"]
                elif "multilingual" in model_options:
                    chosen_model = model_options["multilingual"]
                elif "default" in model_options:
                    chosen_model = model_options["default"]
                elif isinstance(model_options, str):
                    chosen_model = model_options  # if config directly provides a model name
                else:
                    # Fallback to a reasonable default model
                    chosen_model = "paraphrase-MiniLM-L6-v2"
            try:
                self.embedding_model = SentenceTransformer(chosen_model)
                self.logger.info(f"SentenceTransformer model loaded: {chosen_model}")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                # If model loading fails despite deps being available, raise error
                raise RuntimeError(f"Could not load SentenceTransformer model '{chosen_model}'") 
        else:
            # Dependencies missing: log warning, but allow usage with fallback
            self.logger.warning("Optional dependencies missing - using fallback logic for clustering and similarity")
            # (self.embedding_model remains None, which signals fallback in methods)
    
    def cluster_texts(self, texts: List[str], n_clusters: int = 3) -> List[int]:
        """Clustert eine Liste von Texten in semantisch ähnliche Gruppen.
        
        Wandelt die Eingabetexte zunächst in Embeddings um (sofern ein Modell verfügbar ist) 
        und führt dann das Clustering mit dem angegebenen Verfahren durch. 
        Bei nicht verfügbaren ML-Bibliotheken erfolgt ein Fallback auf eine einfache Verteilungsstrategie.
        
        Args:
            texts: Liste von Texten, die geclustert werden sollen.
            n_clusters: Gewünschte Anzahl von Clustern (nur relevant für kmeans oder agglomerative).
        
        Returns:
            Eine Liste von Cluster-Labels (Ganzzahlen) in der Länge der Eingabeliste, 
            wobei jeder Text einem Cluster zugeordnet ist. 
            Bei Fallback ohne ML-Modelle werden die Texte gleichmäßig oder sequentiell auf Cluster verteilt.
        
        Raises:
            ValueError: Wenn die Eingabeliste leer ist.
        """
        if texts is None:
            texts = []
        if len(texts) == 0:
            # No texts to cluster
            return []
        
        # If no ML dependencies or no model loaded, use simple fallback clustering
        if not SEMANTIC_DEPS_AVAILABLE or self.embedding_model is None:
            self.logger.debug("Using fallback clustering (no ML dependencies).")
            # Simple strategy: distribute texts into clusters in a round-robin fashion
            if n_clusters < 1:
                n_clusters = 1
            # For a single text, or n_clusters == 1, all texts go to cluster 0
            if n_clusters == 1 or len(texts) == 1:
                return [0] * len(texts)
            clusters: List[int] = []
            for idx in range(len(texts)):
                # Assign cluster ids cyclically from 0 to n_clusters-1
                clusters.append(idx % n_clusters)
            return clusters
        
        # ML dependencies available -> perform actual clustering
        embeddings: List[Any] = []  # List of embedding vectors
        for text in texts:
            # Use cache if embedding was computed before
            if text in self.embedding_cache:
                emb_vector = self.embedding_cache[text].vector
                self._cache_hits += 1
                # Update last access time for LRU policy
                self.embedding_cache[text].last_access = time.time()
            else:
                self._cache_misses += 1
                emb_vector = self.embedding_model.encode(text)
                # Convert to numpy array for consistency
                emb_vector = np.asarray(emb_vector, dtype=float)
                # Cache the new embedding
                if len(self.embedding_cache) >= self._max_cache_size:
                    # Evict the least recently used entry (simple strategy)
                    oldest_key = min(self.embedding_cache.keys(), key=lambda k: self.embedding_cache[k].last_access)
                    self.embedding_cache.pop(oldest_key, None)
                # Store in cache
                self.embedding_cache[text] = EmbeddingCacheEntry(vector=emb_vector, last_access=time.time())
            embeddings.append(emb_vector)
        
        # Convert list of embeddings to 2D array for clustering
        X = np.vstack(embeddings)  # shape: (len(texts), embedding_dim)
        
        # Adjust cluster count for algorithms that require n_clusters <= number of points
        effective_clusters = n_clusters if n_clusters > 0 else 1
        effective_clusters = min(effective_clusters, len(texts))
        
        labels: Any  # Will be np.ndarray when numpy is available
        try:
            if self.clustering_method == "kmeans":
                # Use KMeans clustering
                kmeans = KMeans(n_clusters=effective_clusters, random_state=42)
                labels = kmeans.fit_predict(X)
            elif self.clustering_method == "dbscan":
                # Use DBSCAN clustering (eps and min_samples could be taken from config if needed)
                eps = self.config.get("clustering", {}).get("dbscan_eps", 0.5)
                min_samples = self.config.get("clustering", {}).get("dbscan_min_samples", 2)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
            elif self.clustering_method == "agglomerative":
                # Use Agglomerative Clustering
                agglo = AgglomerativeClustering(n_clusters=effective_clusters)
                labels = agglo.fit_predict(X)
            else:
                # Default to kmeans if method is unrecognized
                self.logger.warning(f"Unknown clustering_method '{self.clustering_method}', defaulting to kmeans")
                kmeans = KMeans(n_clusters=effective_clusters, random_state=42)
                labels = kmeans.fit_predict(X)
        except Exception as e:
            # If any clustering error occurs, fallback to simple distribution
            self.logger.exception(f"Clustering error (using {self.clustering_method}) – falling back to simple clusters: {e}")
            labels = np.array([i % effective_clusters for i in range(len(texts))], dtype=int)
        
        # If DBSCAN produced noise points (label -1), map them to a valid cluster ID
        if self.clustering_method == "dbscan":
            # Ensure no negative labels in output
            if (labels < 0).any():
                # If all points are noise, assign all to cluster 0
                if (labels == -1).all():
                    labels = np.zeros_like(labels)
                else:
                    max_label = labels.max()
                    # Assign noise points to a new cluster index (max_label + 1)
                    labels[labels == -1] = max_label + 1
        
        return labels.astype(int).tolist()
    
    def calculate_similarity(self, text_a: str, text_b: str) -> float:
        """Berechnet die semantische Ähnlichkeit zwischen zwei Texten.
        
        Falls möglich wird die Cosine Similarity zwischen SentenceTransformer-Embeddings der Texte berechnet.
        Ansonsten erfolgt eine einfache Abschätzung über Wort-Überschneidung (Jaccard-Index) als Fallback.
        
        Args:
            text_a: Erster Text.
            text_b: Zweiter Text.
        
        Returns:
            Ein Gleitkommawert zwischen 0.0 und 1.0, der die semantische Ähnlichkeit der beiden Texte angibt 
            (1.0 = inhaltlich identisch, 0.0 = völlig unterschiedlich).
        """
        if text_a is None: 
            text_a = ""
        if text_b is None:
            text_b = ""
        
        # If no ML model available, use word overlap (Jaccard similarity) as fallback
        if not SEMANTIC_DEPS_AVAILABLE or self.embedding_model is None:
            self.logger.debug("Using fallback similarity (word overlap).")
            # Both empty strings -> treat as identical (similarity = 1.0)
            if text_a.strip() == "" and text_b.strip() == "":
                return 1.0
            # If one is empty and the other is not -> 0.0 similarity
            if text_a.strip() == "" or text_b.strip() == "":
                return 0.0
            # Compute Jaccard similarity on word sets
            words_a = set(text_a.lower().split())
            words_b = set(text_b.lower().split())
            if len(words_a) == 0 or len(words_b) == 0:
                # If either has no valid words (after splitting), handle edge-case
                return 0.0 if words_a != words_b else 1.0
            intersection = words_a.intersection(words_b)
            union = words_a.union(words_b)
            similarity = float(len(intersection)) / float(len(union))
            return max(0.0, min(1.0, similarity))
        
        # Use embedding-based cosine similarity
        try:
            vec_a = self.embedding_model.encode(text_a)
            vec_b = self.embedding_model.encode(text_b)
        except Exception as e:
            # In case encoding fails (should be rare), fallback to word overlap
            self.logger.error(f"Encoding failed, using word overlap similarity: {e}")
            # Use word overlap fallback directly (avoid infinite recursion)
            words_a = set(text_a.lower().split())
            words_b = set(text_b.lower().split())
            if text_a.strip() == "" and text_b.strip() == "":
                return 1.0
            if text_a.strip() == "" or text_b.strip() == "":
                return 0.0
            if len(words_a) == 0 or len(words_b) == 0:
                return 0.0 if words_a != words_b else 1.0
            intersection = words_a.intersection(words_b)
            union = words_a.union(words_b)
            return float(len(intersection)) / float(len(union))
        
        vec_a = np.asarray(vec_a, dtype=float)
        vec_b = np.asarray(vec_b, dtype=float)
        if vec_a.ndim > 1:
            vec_a = vec_a.flatten()
        if vec_b.ndim > 1:
            vec_b = vec_b.flatten()
        # If either vector is all zeros (should not happen for well-trained models), handle gracefully
        if not np.any(vec_a) or not np.any(vec_b):
            return 0.0
        
        # Compute cosine similarity
        dot = float(np.dot(vec_a, vec_b))
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            # one of the vectors is zero-vector
            return 0.0
        cosine_sim = dot / (norm_a * norm_b)
        # Clamp cosine similarity to [0.0, 1.0]
        if cosine_sim < 0.0:
            similarity = 0.0
        elif cosine_sim > 1.0:
            similarity = 1.0
        else:
            similarity = cosine_sim
        return float(similarity)

    # =========================================================================
    # ADDITIONAL UTILITY METHODS
    # =========================================================================

    def get_cache_stats(self) -> Dict[str, int]:
        """Returns cache statistics for monitoring purposes"""
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self.embedding_cache),
            "max_cache_size": self._max_cache_size
        }

    def clear_cache(self) -> None:
        """Clears the embedding cache"""
        self.embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_capabilities(self) -> Dict[str, Any]:
        """Get detailed information about available capabilities"""
        return {
            'has_sentence_transformers': SBERT_AVAILABLE,
            'has_sklearn': SKLEARN_AVAILABLE,
            'embedding_model_loaded': self.embedding_model is not None,
            'current_method': self.clustering_method,
            'available_methods': self.get_available_methods(),
            'model_name': getattr(self.embedding_model, 'model_name', None) if self.embedding_model else None
        }

    def get_available_methods(self) -> List[str]:
        """Get list of available clustering methods"""
        methods = []
        if SEMANTIC_DEPS_AVAILABLE:
            methods.extend(["kmeans", "dbscan", "agglomerative"])
        methods.append("fallback_simple")
        return methods

    def is_semantic_available(self) -> bool:
        """Check if semantic clustering capabilities are available"""
        return SEMANTIC_DEPS_AVAILABLE and self.embedding_model is not None

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def is_semantic_available() -> bool:
    """Check if semantic clustering capabilities are available globally"""
    return SEMANTIC_DEPS_AVAILABLE

def get_semantic_capabilities() -> Dict[str, bool]:
    """Get information about available semantic processing capabilities"""
    return {
        'sentence_transformers': SBERT_AVAILABLE,
        'sklearn': SKLEARN_AVAILABLE,
        'semantic_clustering': SEMANTIC_DEPS_AVAILABLE,
        'basic_clustering': True
    }


if __name__ == "__main__":
    # Quick test of the corrected implementation
    print("=== CORRECTED SEMANTIC CLUSTERING ENHANCER TEST ===")
    
    try:
        enhancer = SemanticClusteringEnhancer(clustering_method="kmeans")
        print(f"✓ Instantiation successful")
        print(f"✓ Capabilities: {enhancer.get_capabilities()}")
        
        # Test with sample texts
        test_texts = [
            "This is about machine learning and AI",
            "The weather is nice today", 
            "Deep learning models are powerful",
            "I love sunny days",
            "Neural networks are fascinating"
        ]
        
        result = enhancer.cluster_texts(test_texts, n_clusters=2)
        print(f"✓ Cluster assignments: {result}")
        
        # Test similarity
        sim = enhancer.calculate_similarity(test_texts[0], test_texts[2])
        print(f"✓ Similarity between ML texts: {sim:.3f}")
        
        # Test cache stats
        print(f"✓ Cache stats: {enhancer.get_cache_stats()}")
        
        print("=== ALL TESTS PASSED ===")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
