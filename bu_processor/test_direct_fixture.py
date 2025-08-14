#!/usr/bin/env python3
"""
Direct fixture test without pytest to isolate the issue
"""

import os
import sys
from pathlib import Path

# Set environment
os.environ["BU_LAZY_MODELS"] = "0"
os.environ["TESTING"] = "true"

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "bu_processor"))
sys.path.insert(0, str(Path(__file__).parent))

def test_direct_fixture():
    """Test fixture functionality directly."""
    try:
        # Import ClassificationResult directly
        from bu_processor.pipeline.classifier import ClassificationResult
        print("✅ ClassificationResult import successful")
        
        # Test creating a result with all required fields
        result = ClassificationResult(
            text="test text",
            category=0,  # Should be an int
            confidence=0.85,
            is_confident=True,
            metadata={"test": True}
        )
        
        # Add extra attributes like the real classifier does
        result.input_type = "text"
        result.text_length = 9
        result.processing_time = 0.001
        result.model_version = "v1.0"
        
        print(f"✅ ClassificationResult created successfully")
        print(f"   Category: {result.category} (type: {type(result.category)})")
        
        # Test the dict conversion like the test does
        if hasattr(result, 'dict'):  # Pydantic model
            result_data = result.dict()
        else:  # Dict
            result_data = result.__dict__
            
        print(f"✅ Dict conversion successful")
        print(f"   Result data keys: {list(result_data.keys())}")
        print(f"   Category in dict: {result_data.get('category')} (type: {type(result_data.get('category'))})")
        
        # Test the assertions from the real test
        assert "category" in result_data
        assert isinstance(result_data["category"], int)
        assert result_data["category"] == 0
        
        print("✅ All fixture assertions passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_fixture()
    exit(0 if success else 1)
