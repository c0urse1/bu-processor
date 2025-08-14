#!/usr/bin/env python3
"""
Debug the module import in detail
"""

import sys
import os
sys.path.insert(0, r'c:\ml_classifier_poc\bu_processor')

print("Testing module import with error tracking...")

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "semantic_chunking_enhancement",
        r"c:\ml_classifier_poc\bu_processor\bu_processor\pipeline\semantic_chunking_enhancement.py"
    )
    module = importlib.util.module_from_spec(spec)
    
    # Capture any errors during execution
    print("Executing module...")
    
    try:
        spec.loader.exec_module(module)
        print("✅ Module execution completed")
    except Exception as e:
        print(f"❌ Error during module execution: {e}")
        import traceback
        traceback.print_exc()
        # Continue anyway to see partial results
    
    print(f"Module attributes: {dir(module)}")
    
    # Check for classes specifically
    classes_found = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type):
            classes_found.append(name)
    
    print(f"Classes found: {classes_found}")
    
    # Check for our specific class
    if hasattr(module, 'SemanticClusteringEnhancer'):
        print("✅ SemanticClusteringEnhancer found!")
        cls = getattr(module, 'SemanticClusteringEnhancer')
        print(f"Class type: {type(cls)}")
        print(f"Class MRO: {cls.__mro__}")
    else:
        print("❌ SemanticClusteringEnhancer not found")
        
        # Check if we have any partial definitions
        for name in dir(module):
            if 'Semantic' in name or 'Clustering' in name:
                print(f"Found related item: {name} = {getattr(module, name)}")
        
except Exception as e:
    print(f"❌ Module loading error: {e}")
    import traceback
    traceback.print_exc()
