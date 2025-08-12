#!/usr/bin/env python3
"""Debug SimHash issues step by step"""

import sys
from pathlib import Path

# Add bu_processor to Python path
script_dir = Path(__file__).resolve().parent
bu_processor_dir = script_dir / "bu_processor"
sys.path.insert(0, str(bu_processor_dir))

print("Starting debug...")

try:
    print("1. Importing SemanticSimHashGenerator...")
    from bu_processor.pipeline.simhash_semantic_deduplication import SemanticSimHashGenerator
    print("✅ Import successful")
    
    print("2. Creating instance...")
    generator = SemanticSimHashGenerator()
    print("✅ Instance created")
    
    print("3. Testing _normalize_text...")
    normalized = generator._normalize_text("  HELLO World!  Test@#$  ")
    print(f"✅ Normalized text: '{normalized}'")
    
    print("4. Testing _extract_features...")
    features = generator._extract_features("hello world test", 2)
    print(f"✅ Features: {features[:3]}...")  # Show first 3 features
    
    print("5. Testing standalone calculate_simhash...")
    from bu_processor.pipeline.simhash_semantic_deduplication import calculate_simhash
    simhash_value = calculate_simhash("This is a test for SimHash calculation")
    print(f"✅ SimHash value: {simhash_value}")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
