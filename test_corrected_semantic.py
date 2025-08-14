#!/usr/bin/env python3

print("Testing corrected SemanticClusteringEnhancer...")

try:
    from semantic_clustering_compatible import SemanticClusteringEnhancer
    print("✓ Import successful")
    
    enhancer = SemanticClusteringEnhancer()
    print("✓ Instantiation successful")
    
    caps = enhancer.get_capabilities()
    print(f"✓ Capabilities: {caps}")
    
    # Test clustering
    texts = ["Hello world", "Goodbye moon", "Testing cluster"]
    result = enhancer.cluster_texts(texts, n_clusters=2)
    print(f"✓ Clustering: {result}")
    
    # Test similarity
    sim = enhancer.calculate_similarity("Hello world", "Hello earth")
    print(f"✓ Similarity: {sim}")
    
    print("=== SUCCESS: Corrected implementation working! ===")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
