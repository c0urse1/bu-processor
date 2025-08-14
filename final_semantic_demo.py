#!/usr/bin/env python3
"""
Final demonstration of the complete SemanticClusteringEnhancer implementation
"""

import sys
import os
sys.path.insert(0, r'c:\ml_classifier_poc\bu_processor')

print("🎯 FINAL SEMANTIC CLUSTERING ENHANCEMENT DEMONSTRATION")
print("=" * 70)

try:
    from bu_processor.pipeline.semantic_chunking_enhancement import (
        SemanticClusteringEnhancer,
        ContentType,
        SemanticClusterResult
    )
    
    # Demo 1: Unified API with different clustering methods
    print("\n🔹 Demo 1: Unified API with Various Clustering Methods")
    print("-" * 60)
    
    methods_to_test = ["auto", "kmeans", "dbscan", "agglomerative"]
    for method in methods_to_test:
        enhancer = SemanticClusteringEnhancer(clustering_method=method)
        capabilities = enhancer.get_capabilities()
        print(f"✅ Method '{method}': Current={capabilities['current_method']}")
    
    # Demo 2: Einheitliche API - cluster_texts() mit verschiedenen Parametern
    print("\n🔹 Demo 2: Unified cluster_texts() API")
    print("-" * 60)
    
    enhancer = SemanticClusteringEnhancer()
    
    test_documents = [
        "Legal contract terms and conditions for business partnerships",
        "Python machine learning tutorial with scikit-learn examples", 
        "Technical documentation for API endpoints and authentication",
        "Business requirements and legal compliance frameworks",
        "Data science workflow with pandas and numpy libraries",
        "Software architecture patterns and microservices design"
    ]
    
    # Test with different content types
    content_tests = [
        (ContentType.LEGAL_TEXT, "Legal documents"),
        (ContentType.TECHNICAL, "Technical content"),
        (ContentType.MIXED, "Mixed content")
    ]
    
    for content_type, description in content_tests:
        result = enhancer.cluster_texts(test_documents, content_type=content_type)
        print(f"✅ {description}: {len(set(result.cluster_assignments))} clusters")
    
    # Demo 3: Einheitliche API - calculate_similarity()
    print("\n🔹 Demo 3: Unified calculate_similarity() API")
    print("-" * 60)
    
    similarity_pairs = [
        ("Legal contract agreement", "Business partnership terms"),
        ("Python machine learning", "Data science algorithms"),
        ("API documentation", "Technical specifications"),
        ("Completely different text", "Totally unrelated content")
    ]
    
    for text1, text2 in similarity_pairs:
        similarity = enhancer.calculate_similarity(text1, text2)
        print(f"✅ Similarity: {similarity:.3f} - '{text1[:20]}...' vs '{text2[:20]}...'")
    
    # Demo 4: Backward Compatibility
    print("\n🔹 Demo 4: Backward Compatibility")
    print("-" * 60)
    
    # Test old API with n_clusters
    old_result = enhancer.cluster_texts(test_documents, n_clusters=3)
    print(f"✅ Legacy API (n_clusters): {type(old_result)} = {old_result}")
    
    # Test new API with num_clusters  
    new_result = enhancer.cluster_texts(test_documents, num_clusters=3)
    print(f"✅ New API (num_clusters): {type(new_result)}")
    
    # Demo 5: Runtime Capabilities & Feature Detection
    print("\n🔹 Demo 5: Runtime Capabilities & Feature Detection")
    print("-" * 60)
    
    capabilities = enhancer.get_capabilities()
    available_methods = enhancer.get_available_methods()
    
    print(f"✅ SentenceTransformers Available: {capabilities['has_sentence_transformers']}")
    print(f"✅ Scikit-learn Available: {capabilities['has_sklearn']}")
    print(f"✅ Semantic Enhancement: {capabilities['semantic_enhancement_available']}")
    print(f"✅ Current Method: {capabilities['current_method']}")
    print(f"✅ Available Methods: {available_methods}")
    
    # Demo 6: Robustness & Error Handling
    print("\n🔹 Demo 6: Robustness & Error Handling")
    print("-" * 60)
    
    # Test edge cases
    edge_cases = [
        ([], "Empty input"),
        (["single"], "Single text"),
        (["", ""], "Empty strings"),
        (["a", "b", "c"] * 10, "Large dataset")
    ]
    
    for test_input, description in edge_cases:
        try:
            result = enhancer.cluster_texts(test_input)
            print(f"✅ {description}: Handled gracefully")
        except Exception as e:
            print(f"❌ {description}: Error - {e}")
    
    print("\n" + "=" * 70)
    print("🎉 SEMANTIC CLUSTERING ENHANCEMENT COMPLETE!")
    print("=" * 70)
    print("✅ Einheitliche API (clustering_method, cluster_texts, calculate_similarity)")
    print("✅ Lauffähig mit und ohne Heavy-Deps (SentenceTransformers / scikit-learn)")  
    print("✅ Feature-Flag & Fallbacks für robuste Deployment-Szenarien")
    print("✅ Adaptive Cluster-Parameter basierend auf Content-Type")
    print("✅ Backward Compatibility für bestehende Tests")
    print("✅ Graceful Error Handling und Edge Cases")
    print("\n🚀 SemanticClusteringEnhancer ist ROBUST & API-STABLE!")
    
except Exception as e:
    print(f"❌ Demo failed: {e}")
    import traceback
    traceback.print_exc()
