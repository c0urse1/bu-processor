#!/usr/bin/env python3
"""Simple test for SimHash class instantiation"""

try:
    from bu_processor.bu_processor.pipeline.simhash_semantic_deduplication import SemanticSimHashGenerator
    print("✓ Import successful")
    
    generator = SemanticSimHashGenerator()
    print("✓ SemanticSimHashGenerator created successfully")
    
    # Test basic functionality
    test_text = "This is a test document for SimHash functionality."
    hash_value = generator.calculate_simhash(test_text)
    print(f"✓ SimHash calculated: {hash_value}")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
