#!/usr/bin/env python3
"""
Debug the semantic_chunking_enhancement module
"""
import sys
import traceback

try:
    print("Attempting to import semantic_chunking_enhancement...")
    import bu_processor.pipeline.semantic_chunking_enhancement as sce
    print(f"Module imported successfully: {sce}")
    print(f"Module file: {sce.__file__}")
    print(f"Available attributes: {[attr for attr in dir(sce) if not attr.startswith('_')]}")
    
    # Try to access the class directly
    if hasattr(sce, 'SemanticClusteringEnhancer'):
        print("✅ SemanticClusteringEnhancer found!")
        enhancer = sce.SemanticClusteringEnhancer()
        print("✅ Instance created successfully!")
    else:
        print("❌ SemanticClusteringEnhancer not found in module")
        
except Exception as e:
    print(f"❌ Error during import: {e}")
    traceback.print_exc()
