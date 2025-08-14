#!/usr/bin/env python3
"""
Test the SemanticClusteringEnhancer class directly
"""

import sys
import os
sys.path.insert(0, r'c:\ml_classifier_poc\bu_processor')

print("Testing direct import...")

try:
    # Import directly without going through bu_processor package
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "semantic_chunking_enhancement",
        r"c:\ml_classifier_poc\bu_processor\bu_processor\pipeline\semantic_chunking_enhancement.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    print("✅ Module loaded successfully")
    print(f"Available attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}")
    
    if hasattr(module, 'SemanticClusteringEnhancer'):
        print("✅ SemanticClusteringEnhancer found!")
        
        # Test basic instantiation
        enhancer = module.SemanticClusteringEnhancer()
        print("✅ Instance created successfully!")
        
        # Test unified API methods
        capabilities = enhancer.get_capabilities()
        print(f"✅ Capabilities: {capabilities}")
        
        # Test simple clustering
        test_texts = ["Hello world", "Python programming", "Machine learning", "Data science"]
        result = enhancer.cluster_texts(test_texts, num_clusters=2)
        print(f"✅ Clustering result: {result}")
        
        # Test similarity calculation
        similarity = enhancer.calculate_similarity("Hello world", "Hi there")
        print(f"✅ Similarity score: {similarity}")
        
        print("🎉 All tests passed!")
        
    else:
        print("❌ SemanticClusteringEnhancer not found in module")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
