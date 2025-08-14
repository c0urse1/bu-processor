#!/usr/bin/env python3
"""
Simple test script for SemanticClusteringEnhancer
"""

import sys
import os
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent / "bu_processor"))

def main():
    print("=== Simple SemanticClusteringEnhancer Test ===")
    
    # Set testing environment
    os.environ["BU_LAZY_MODELS"] = "1"
    os.environ["TESTING"] = "true"
    
    try:
        print("Importing module...")
        import bu_processor.pipeline.semantic_chunking_enhancement as sce
        print("✅ Module import successful")
        
        print("Importing class...")
        from bu_processor.pipeline.semantic_chunking_enhancement import SemanticClusteringEnhancer
        print("✅ Class import successful")
        
        print("Creating instance...")
        enhancer = SemanticClusteringEnhancer()
        print("✅ Instance creation successful")
        
        print("Testing basic functionality...")
        # Test feature availability
        features = enhancer.get_available_features()
        print(f"✅ Available features: {features}")
        
        # Test clustering with simple texts
        sample_texts = ["Hello world", "Python programming", "Machine learning"]
        clusters = enhancer.cluster_texts(sample_texts)
        print(f"✅ Clustering result: {clusters}")
        
        # Test similarity
        similarity = enhancer.calculate_similarity("Hello", "Hi")
        print(f"✅ Similarity calculation: {similarity}")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
