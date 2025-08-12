#!/usr/bin/env python3
"""Test semantic enhancer consistency fixes - Fix #8"""

import sys
from pathlib import Path

# Add project root to Python path  
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
# Also add bu_processor specifically
bu_processor_dir = script_dir / "bu_processor"
sys.path.insert(0, str(bu_processor_dir))

def test_semantic_enhancer_fixes():
    """Test dass die Semantic-Enhancer Fixes korrekt implementiert wurden."""
    
    print("🔍 Testing Semantic Enhancer Consistency Fixes...")
    
    try:
        # Test 1: Import SemanticClusteringEnhancer
        from bu_processor.pipeline.semantic_chunking_enhancement import SemanticClusteringEnhancer
        print("✅ SemanticClusteringEnhancer import successful")
        
        # Test 2: Check clustering_method parameter exists
        enhancer = SemanticClusteringEnhancer(clustering_method="kmeans")
        assert hasattr(enhancer, 'clustering_method'), "clustering_method attribute missing"
        assert enhancer.clustering_method == "kmeans", "clustering_method not set correctly"
        print("✅ clustering_method parameter works correctly")
        
        # Test 3: Check cluster_texts method exists
        assert hasattr(enhancer, 'cluster_texts'), "cluster_texts method missing"
        assert callable(getattr(enhancer, 'cluster_texts')), "cluster_texts is not callable"
        print("✅ cluster_texts method exists and is callable")
        
        # Test 4: Check calculate_similarity method exists
        assert hasattr(enhancer, 'calculate_similarity'), "calculate_similarity method missing"
        assert callable(getattr(enhancer, 'calculate_similarity')), "calculate_similarity is not callable"
        print("✅ calculate_similarity method exists and is callable")
        
        # Test 5: Test cluster_texts with fallback (no dependencies)
        test_texts = ["Text A", "Text B", "Text C"]
        clusters = enhancer.cluster_texts(test_texts, n_clusters=2)
        assert isinstance(clusters, list), "cluster_texts should return a list"
        assert len(clusters) == len(test_texts), "cluster_texts should return same length as input"
        print("✅ cluster_texts fallback logic works")
        
        # Test 6: Test calculate_similarity with fallback
        similarity = enhancer.calculate_similarity("Hello world", "Hello earth")
        assert isinstance(similarity, float), "calculate_similarity should return float"
        assert 0.0 <= similarity <= 1.0, "similarity should be between 0 and 1"
        print("✅ calculate_similarity fallback logic works")
        
        # Test 7: Test different clustering methods
        for method in ["kmeans", "dbscan", "agglomerative"]:
            enhancer_method = SemanticClusteringEnhancer(clustering_method=method)
            assert enhancer_method.clustering_method == method, f"clustering_method {method} not set correctly"
        print("✅ All clustering methods can be set")
        
        print("\n🎉 All Semantic Enhancer fixes working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing semantic enhancer: {e}")
        return False

def test_graceful_dependency_handling():
    """Test dass Dependency-Handling gracefully funktioniert."""
    
    print("\n🔍 Testing Graceful Dependency Handling...")
    
    try:
        from bu_processor.pipeline.semantic_chunking_enhancement import SemanticClusteringEnhancer
        
        # Test ohne sentence-transformers/sklearn - sollte fallback verwenden
        enhancer = SemanticClusteringEnhancer()
        
        # Diese calls sollten nicht crashen, auch ohne dependencies
        test_texts = ["Test document one", "Test document two"]
        clusters = enhancer.cluster_texts(test_texts, n_clusters=2)
        similarity = enhancer.calculate_similarity("text1", "text2")
        
        print("✅ Graceful fallback works without dependencies")
        return True
        
    except Exception as e:
        print(f"❌ Error with dependency handling: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Semantic Enhancer Consistency Fixes (Fix #8)")
    print("=" * 60)
    
    success = True
    
    # Test the main fixes
    success &= test_semantic_enhancer_fixes()
    
    # Test dependency handling
    success &= test_graceful_dependency_handling()
    
    print("\n" + "=" * 60)
    if success:
        print("🎯 ALL SEMANTIC ENHANCER TESTS PASSED!")
        print("✅ Fix #8: Semantic‑Enhancer / Methoden & Parameter konsistent - COMPLETED")
    else:
        print("❌ Some tests failed!")
    
    sys.exit(0 if success else 1)
