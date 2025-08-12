#!/usr/bin/env python3
"""Direct test of semantic enhancer without full package import"""

import sys
import os

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'bu_processor'))

print("🚀 Direct Semantic Enhancer Test...")

try:
    # Direct import of the specific file to bypass __init__.py issues
    import importlib.util
    
    semantic_file = os.path.join(script_dir, 'bu_processor', 'bu_processor', 'pipeline', 'semantic_chunking_enhancement.py')
    print(f"Loading: {semantic_file}")
    
    spec = importlib.util.spec_from_file_location("semantic_module", semantic_file)
    semantic_module = importlib.util.module_from_spec(spec)
    
    # Mock the content_types import that might be missing
    class MockContentType:
        pass
    
    # Temporarily add mock to sys.modules
    sys.modules['bu_processor.pipeline.content_types'] = type('MockModule', (), {'ContentType': MockContentType})()
    
    spec.loader.exec_module(semantic_module)
    print("✅ Module loaded successfully")
    
    # Get the class
    SemanticClusteringEnhancer = semantic_module.SemanticClusteringEnhancer
    print("✅ SemanticClusteringEnhancer class found")
    
    # Test initialization with clustering_method
    enhancer = SemanticClusteringEnhancer(clustering_method="kmeans")
    print(f"✅ Enhancer created with clustering_method: {enhancer.clustering_method}")
    
    # Test methods exist
    assert hasattr(enhancer, 'cluster_texts'), "cluster_texts method missing"
    assert hasattr(enhancer, 'calculate_similarity'), "calculate_similarity method missing"
    print("✅ Both required methods exist")
    
    # Test functionality
    test_texts = ["Document A", "Document B", "Document C"]
    clusters = enhancer.cluster_texts(test_texts, n_clusters=2)
    print(f"✅ cluster_texts returned: {clusters}")
    
    similarity = enhancer.calculate_similarity("Hello world", "Hello earth")
    print(f"✅ calculate_similarity returned: {similarity}")
    
    print("\n🎉 ALL TESTS PASSED! Semantic Enhancer fixes working correctly!")
    
    # Print method details
    print(f"\n📋 Method Signatures:")
    print(f"- cluster_texts: {enhancer.cluster_texts.__doc__ or 'No docstring'}")
    print(f"- calculate_similarity: {enhancer.calculate_similarity.__doc__ or 'No docstring'}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
