#!/usr/bin/env python3

import sys
sys.path.insert(0, r'c:\ml_classifier_poc')

print("Testing import...")

try:
    from bu_processor.bu_processor.pipeline.semantic_chunking_enhancement import SemanticClusteringEnhancer
    print("✓ Import successful!")
    
    enhancer = SemanticClusteringEnhancer()
    print("✓ Class instantiated!")
    
    print(f"✓ Capabilities: {enhancer.get_capabilities()}")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
