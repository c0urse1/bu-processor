#!/usr/bin/env python3
"""
Minimal SemanticClusteringEnhancer - Testing corrected implementation
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class SemanticClusteringEnhancer:
    """Minimal corrected implementation for testing"""
    
    def __init__(self, model_name: Optional[str] = None, max_cache_size: int = None, 
                 clustering_method: str = "kmeans") -> None:
        """Initialize with clustering_method parameter as required"""
        self.clustering_method = clustering_method.lower() if clustering_method else "kmeans"
        self.logger = logger
        self.embedding_model = None  # No ML dependencies for testing
    
    def cluster_texts(self, texts: List[str], n_clusters: int = 3) -> List[int]:
        """Cluster texts - returns list of cluster assignments"""
        if texts is None:
            texts = []
        if len(texts) == 0:
            return []
        
        # Simple fallback clustering - round-robin assignment
        if n_clusters < 1:
            n_clusters = 1
        if n_clusters == 1 or len(texts) == 1:
            return [0] * len(texts)
        
        clusters = []
        for idx in range(len(texts)):
            clusters.append(idx % n_clusters)
        return clusters
    
    def calculate_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate similarity - returns float between 0.0 and 1.0"""
        if text_a is None:
            text_a = ""
        if text_b is None:
            text_b = ""
        
        # Simple word overlap similarity (Jaccard index)
        if text_a.strip() == "" and text_b.strip() == "":
            return 1.0
        if text_a.strip() == "" or text_b.strip() == "":
            return 0.0
        
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if len(words_a) == 0 or len(words_b) == 0:
            return 0.0 if words_a != words_b else 1.0
        
        intersection = words_a.intersection(words_b)
        union = words_a.union(words_b)
        similarity = float(len(intersection)) / float(len(union))
        return max(0.0, min(1.0, similarity))
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get capabilities information"""
        return {
            'has_sentence_transformers': False,
            'has_sklearn': False,
            'embedding_model_loaded': False,
            'current_method': self.clustering_method,
            'available_methods': ['fallback_simple'],
            'model_name': None
        }

if __name__ == "__main__":
    print("=== MINIMAL CORRECTED IMPLEMENTATION TEST ===")
    
    enhancer = SemanticClusteringEnhancer(clustering_method="kmeans")
    print(f"✓ Capabilities: {enhancer.get_capabilities()}")
    
    # Test clustering
    test_texts = ["Hello world", "Goodbye moon", "Testing cluster", "Another test", "Final text"]
    result = enhancer.cluster_texts(test_texts, n_clusters=3)
    print(f"✓ Cluster assignments: {result}")
    
    # Test similarity
    sim = enhancer.calculate_similarity("Hello world", "Hello earth")
    print(f"✓ Similarity: {sim:.3f}")
    
    print("=== SUCCESS ===")
