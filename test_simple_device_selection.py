#!/usr/bin/env python3
"""
Simple test for the _pick_device function.
"""

import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_pick_device_function():
    """Test the _pick_device function in isolation."""
    print("Testing _pick_device function...")
    
    # Test 1: No torch available
    print("\n1. Testing without torch:")
    with patch.dict(sys.modules, {'torch': None}):
        # Need to reload the module to pick up the None torch
        if 'bu_processor.bu_processor.pipeline.classifier' in sys.modules:
            del sys.modules['bu_processor.bu_processor.pipeline.classifier']
        from bu_processor.bu_processor.pipeline.classifier import _pick_device
        
        result = _pick_device(True)
        print(f"   use_gpu=True, no torch -> {result}")
        assert result == "cpu", f"Expected 'cpu', got '{result}'"
        
        result = _pick_device(False)
        print(f"   use_gpu=False, no torch -> {result}")
        assert result == "cpu", f"Expected 'cpu', got '{result}'"
        print("   ✅ Correctly handles missing torch")
    
    # Test 2: Torch available, CUDA available
    print("\n2. Testing with torch and CUDA:")
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    
    with patch.dict(sys.modules, {'torch': mock_torch}):
        # Need to reload the module to pick up the mock torch
        if 'bu_processor.bu_processor.pipeline.classifier' in sys.modules:
            del sys.modules['bu_processor.bu_processor.pipeline.classifier']
        from bu_processor.bu_processor.pipeline.classifier import _pick_device
        
        result = _pick_device(True)
        print(f"   use_gpu=True, CUDA available -> {result}")
        assert result == "cuda", f"Expected 'cuda', got '{result}'"
        
        result = _pick_device(False)
        print(f"   use_gpu=False, CUDA available -> {result}")
        assert result == "cpu", f"Expected 'cpu', got '{result}'"
        print("   ✅ Correctly handles GPU preference with CUDA")
    
    # Test 3: Torch available, CUDA not available
    print("\n3. Testing with torch but no CUDA:")
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    
    with patch.dict(sys.modules, {'torch': mock_torch}):
        # Need to reload the module to pick up the updated mock
        if 'bu_processor.bu_processor.pipeline.classifier' in sys.modules:
            del sys.modules['bu_processor.bu_processor.pipeline.classifier']
        from bu_processor.bu_processor.pipeline.classifier import _pick_device
        
        result = _pick_device(True)
        print(f"   use_gpu=True, CUDA not available -> {result}")
        assert result == "cpu", f"Expected 'cpu', got '{result}'"
        
        result = _pick_device(False)
        print(f"   use_gpu=False, CUDA not available -> {result}")
        assert result == "cpu", f"Expected 'cpu', got '{result}'"
        print("   ✅ Correctly falls back to CPU when CUDA unavailable")

if __name__ == "__main__":
    try:
        test_pick_device_function()
        print("\n🎉 All device selection tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
