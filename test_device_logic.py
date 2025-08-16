#!/usr/bin/env python3
"""
Test the device selection logic in isolation by copying the function.
"""

from unittest.mock import MagicMock

def _pick_device(use_gpu, torch_module):
    """
    Test version of device selection logic.
    """
    if use_gpu and torch_module is not None:
        try:
            if torch_module.cuda.is_available():
                return "cuda"
        except Exception:
            pass  # Fall back to CPU if any issue
    return "cpu"

def test_device_logic():
    """Test device selection logic directly."""
    print("Testing device selection logic...")
    
    # Test 1: No torch
    print("\n1. No torch module:")
    result = _pick_device(True, None)
    print(f"   use_gpu=True, torch=None -> {result}")
    assert result == "cpu"
    
    result = _pick_device(False, None)
    print(f"   use_gpu=False, torch=None -> {result}")
    assert result == "cpu"
    print("   ✅ Correctly handles missing torch")
    
    # Test 2: Torch with CUDA
    print("\n2. Torch with CUDA available:")
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    
    result = _pick_device(True, mock_torch)
    print(f"   use_gpu=True, CUDA available -> {result}")
    assert result == "cuda"
    
    result = _pick_device(False, mock_torch)
    print(f"   use_gpu=False, CUDA available -> {result}")
    assert result == "cpu"
    print("   ✅ Correctly respects GPU preference with CUDA")
    
    # Test 3: Torch without CUDA
    print("\n3. Torch without CUDA:")
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    
    result = _pick_device(True, mock_torch)
    print(f"   use_gpu=True, CUDA not available -> {result}")
    assert result == "cpu"
    
    result = _pick_device(False, mock_torch)
    print(f"   use_gpu=False, CUDA not available -> {result}")
    assert result == "cpu"
    print("   ✅ Correctly falls back to CPU when CUDA unavailable")
    
    # Test 4: Torch with broken CUDA
    print("\n4. Torch with broken CUDA check:")
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.side_effect = Exception("CUDA error")
    
    result = _pick_device(True, mock_torch)
    print(f"   use_gpu=True, CUDA check fails -> {result}")
    assert result == "cpu"
    print("   ✅ Correctly handles CUDA check exceptions")

if __name__ == "__main__":
    try:
        test_device_logic()
        print("\n🎉 All device selection logic tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
