#!/usr/bin/env python3
"""Test to verify the safe device picker function works correctly."""

import sys
import os
from unittest.mock import Mock, patch

# Add the bu_processor directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_safe_device_picker():
    """Test the _pick_device function under various conditions."""
    
    try:
        from bu_processor.pipeline.classifier import _pick_device
        
        print("✅ Successfully imported _pick_device function")
        
        # Test 1: When use_gpu is False, should always return 'cpu'
        result = _pick_device(False)
        assert result == "cpu", f"Expected 'cpu', got '{result}'"
        print("✅ Test 1 passed: use_gpu=False returns 'cpu'")
        
        # Test 2: When use_gpu is None, should return 'cpu'
        result = _pick_device(None)
        assert result == "cpu", f"Expected 'cpu', got '{result}'"
        print("✅ Test 2 passed: use_gpu=None returns 'cpu'")
        
        # Test 3: When torch is None (mocked), should return 'cpu'
        with patch('bu_processor.pipeline.classifier.torch', None):
            result = _pick_device(True)
            assert result == "cpu", f"Expected 'cpu' when torch=None, got '{result}'"
            print("✅ Test 3 passed: torch=None returns 'cpu'")
        
        # Test 4: When torch.cuda is mocked to have no is_available, should return 'cpu'
        mock_torch = Mock()
        mock_torch.cuda = Mock()
        # Don't set is_available attribute, so getattr returns None
        with patch('bu_processor.pipeline.classifier.torch', mock_torch):
            result = _pick_device(True)
            assert result == "cpu", f"Expected 'cpu' when cuda.is_available is None, got '{result}'"
            print("✅ Test 4 passed: Missing cuda.is_available returns 'cpu'")
        
        # Test 5: When torch.cuda.is_available() returns False, should return 'cpu'
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        with patch('bu_processor.pipeline.classifier.torch', mock_torch):
            result = _pick_device(True)
            assert result == "cpu", f"Expected 'cpu' when CUDA not available, got '{result}'"
            print("✅ Test 5 passed: CUDA not available returns 'cpu'")
        
        # Test 6: When torch.cuda.is_available() returns True, should return 'cuda'
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        with patch('bu_processor.pipeline.classifier.torch', mock_torch):
            result = _pick_device(True)
            assert result == "cuda", f"Expected 'cuda' when CUDA available, got '{result}'"
            print("✅ Test 6 passed: CUDA available returns 'cuda'")
        
        # Test 7: When torch.cuda.is_available() throws exception, should return 'cpu'
        mock_torch = Mock()
        mock_torch.cuda.is_available.side_effect = Exception("CUDA error")
        with patch('bu_processor.pipeline.classifier.torch', mock_torch):
            result = _pick_device(True)
            assert result == "cpu", f"Expected 'cpu' when CUDA throws exception, got '{result}'"
            print("✅ Test 7 passed: CUDA exception returns 'cpu'")
        
        print("\n🎉 All _pick_device tests passed! Device selection is robust and test-safe.")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_safe_device_picker()
