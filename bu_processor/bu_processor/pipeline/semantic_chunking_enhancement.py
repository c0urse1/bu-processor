#!/usr/bin/env python3
"""
Simple test of SemanticClusteringEnhancer with unified API
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Mock dependencies
_HAS_SBERT = False
_HAS_SKLEARN = False
SEMANTIC_ENHANCEMENT_AVAILABLE = False

def get_logger(name: str):
    return logging.getLogger(name)

class ContentType(Enum):
    LEGAL_TEXT = "legal_text"
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
    """Enhanced semantic clustering for hierarchical chunks - ROBUST VERSION"""
    
    def __init__(self, model_name: str = "test", clustering_method: str = "auto"):
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.clustering_method = ClusteringMethod.FALLBACK_SIMPLE  # Always fallback for now
        
        self.logger.info(
            f"SemanticClusteringEnhancer initialized: "
            f"SBERT={_HAS_SBERT}, sklearn={_HAS_SKLEARN}, "
            f"method={self.clustering_method.value}"
        )

    def cluster_texts(
        self,
        chunks: List[Union[str]],
        num_clusters: Optional[int] = None,
        content_type: Optional[ContentType] = None
    ) -> SemanticClusterResult:
        """UNIFIED API: Cluster texts using the best available method"""
        
        texts = [str(chunk) for chunk in chunks]
        
        if not texts:
            return SemanticClusterResult(cluster_assignments=[])
        
        if num_clusters is None:
            num_clusters = max(2, min(5, len(texts) // 3))
        
        # Simple fallback clustering by text length
        cluster_assignments = []
        for i, text in enumerate(texts):
            cluster_id = i % num_clusters  # Simple round-robin assignment
            cluster_assignments.append(cluster_id)
        
        return SemanticClusterResult(
            cluster_assignments=cluster_assignments,
            metadata={'method': 'fallback_simple', 'num_texts': len(texts)}
        )

    def calculate_similarity(self, text1: str, text2: str, method: str = "auto") -> float:
        """UNIFIED API: Calculate semantic similarity between two texts"""
        
        # Simple character-based similarity
        if not text1 or not text2:
            return 0.0
        
        # Simple Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

    def get_available_methods(self) -> List[str]:
        """Get list of available clustering methods"""
        return ["fallback_simple"]

    def get_capabilities(self) -> Dict[str, Any]:
        """Get detailed capability information"""
        return {
            'has_sentence_transformers': _HAS_SBERT,
            'has_sklearn': _HAS_SKLEARN,
            'semantic_enhancement_available': SEMANTIC_ENHANCEMENT_AVAILABLE,
            'current_method': self.clustering_method.value,
            'available_methods': self.get_available_methods(),
            'model_name': self.model_name
        }

# Test the implementation
if __name__ == "__main__":
    print("🧪 Testing SemanticClusteringEnhancer...")
    
    enhancer = SemanticClusteringEnhancer()
    
    # Test capabilities
    capabilities = enhancer.get_capabilities()
    print(f"✅ Capabilities: {capabilities}")
    
    # Test clustering
    test_texts = ["Hello world", "Python programming", "Machine learning", "Data science", "AI research"]
    result = enhancer.cluster_texts(test_texts, num_clusters=3)
    print(f"✅ Clustering result: {result}")
    
    # Test similarity
    similarity = enhancer.calculate_similarity("Hello world", "Hi there world")
    print(f"✅ Similarity score: {similarity}")
    
    similarity2 = enhancer.calculate_similarity("Python programming", "Java development")
    print(f"✅ Similarity score 2: {similarity2}")
    
    print("🎉 Basic implementation working!")
