#!/usr/bin/env python3
"""
Test that category field works as int
"""

import os
import sys
from pathlib import Path

# Set environment
os.environ["BU_LAZY_MODELS"] = "0"
os.environ["TESTING"] = "true"

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "bu_processor"))

def test_category_int():
    """Test that category accepts int now."""
    try:
        from bu_processor.pipeline.classifier import ClassificationResult
        
        # Test with int category
        result = ClassificationResult(
            text="test text",
            category=0,  # int
            confidence=0.85,
            is_confident=True,
            metadata={"test": True}
        )
        
        print(f"✅ ClassificationResult with int category: {result.category}")
        print(f"   Category type: {type(result.category)}")
        
        # Test dict conversion
        result_dict = result.dict()
        print(f"✅ Dict conversion successful")
        print(f"   Category in dict: {result_dict['category']} (type: {type(result_dict['category'])})")
        
        # Verify it's still an int
        assert isinstance(result_dict["category"], int)
        assert result_dict["category"] == 0
        
        print("✅ Category int validation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_category_int()
    exit(0 if success else 1)
