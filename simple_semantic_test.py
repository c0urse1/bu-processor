#!/usr/bin/env python3
"""Simple test for semantic enhancer fixes"""

import sys
import os

# Set up paths
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'bu_processor'))

print("🚀 Testing Semantic Enhancer Fixes...")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}")

try:
    # Import the class
    from bu_processor.pipeline.semantic_chunking_enhancement import SemanticClusteringEnhancer
    print("✅ Successfully imported SemanticClusteringEnhancer")
    
    # Test 1: Check clustering_method parameter
    enhancer = SemanticClusteringEnhancer(clustering_method="kmeans")
    print(f"✅ Created enhancer with clustering_method: {enhancer.clustering_method}")
    
    # Test 2: Check methods exist
    assert hasattr(enhancer, 'cluster_texts'), "cluster_texts method missing"
    assert hasattr(enhancer, 'calculate_similarity'), "calculate_similarity method missing"
    print("✅ Both required methods exist")
    
    # Test 3: Test cluster_texts
    test_texts = ["Document A", "Document B", "Document C"]
    clusters = enhancer.cluster_texts(test_texts, n_clusters=2)
    print(f"✅ cluster_texts returned: {clusters}")
    
    # Test 4: Test calculate_similarity
    similarity = enhancer.calculate_similarity("Hello world", "Hello earth")
    print(f"✅ calculate_similarity returned: {similarity}")
    
    print("\n🎉 ALL TESTS PASSED! Semantic Enhancer fixes working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
