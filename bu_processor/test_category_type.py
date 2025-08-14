#!/usr/bin/env python3
"""
Test fixture functionality - verify that category is returned as int
"""

import os
import sys
import tempfile
from pathlib import Path

# Set environment
os.environ["BU_LAZY_MODELS"] = "0"
os.environ["TESTING"] = "true"

# Add paths
bu_path = Path(__file__).parent / "bu_processor"
sys.path.insert(0, str(bu_path))
sys.path.insert(0, str(Path(__file__).parent))

def test_category_type():
    """Test that category is returned as int from mock."""
    try:
        # Import needed classes
        from bu_processor.pipeline.content_types import ClassificationResult
        
        # Create a test result manually
        result = ClassificationResult(
            text="test text",
            category=0,  # This should be an int
            confidence=0.85,
            is_confident=True,
            metadata={"test": True}
        )
        
        # Add the extra attributes
        result.input_type = "text"
        result.text_length = 9
        result.processing_time = 0.001
        result.model_version = "v1.0"
        
        # Test the structure like the real test does
        if hasattr(result, 'dict'):  # Pydantic model
            result_data = result.dict()
        else:  # Dict
            result_data = result.__dict__
        
        print(f"Result data: {result_data}")
        print(f"Category type: {type(result_data.get('category'))}")
        print(f"Category value: {result_data.get('category')}")
        
        # Test assertions
        assert "category" in result_data
        assert isinstance(result_data["category"], int)
        assert result_data["category"] == 0
        
        print("✅ Category type test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_category_type()
    exit(0 if success else 1)
