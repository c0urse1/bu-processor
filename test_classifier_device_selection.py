#!/usr/bin/env python3
"""
Test script for classifier device selection functionality.
Tests both CPU-only fallback and config-driven GPU selection.
"""

import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_device_selection_no_torch():
    """Test that device selection gracefully falls back when torch is unavailable."""
    print("=== Test 1: Device selection without torch ===")
    
    # Mock torch as unavailable before importing
    with patch.dict(sys.modules, {'torch': None}):
        # Import the _pick_device function
        from bu_processor.bu_processor.pipeline.classifier import _pick_device
        
        result = _pick_device(use_gpu=True)
        assert result == "cpu", f"Expected 'cpu', got '{result}'"
        print("✅ Correctly falls back to CPU when torch unavailable")
    
    print()

def test_device_selection_with_config():
    """Test that device selection respects config settings."""
    print("=== Test 2: Device selection with config ===")
    
    # Mock torch module
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    
    with patch.dict(sys.modules, {'torch': mock_torch}):
        from bu_processor.bu_processor.pipeline.classifier import _pick_device
        
        # Test GPU enabled
        result = _pick_device(use_gpu=True)
        assert result == "cuda", f"Expected 'cuda', got '{result}'"
        print("✅ Correctly selects CUDA when GPU enabled and available")
        
        # Test GPU disabled
        result = _pick_device(use_gpu=False)
        assert result == "cpu", f"Expected 'cpu', got '{result}'"
        print("✅ Correctly selects CPU when GPU disabled")
        
        # Test GPU enabled but not available
        mock_torch.cuda.is_available.return_value = False
        result = _pick_device(use_gpu=True)
        assert result == "cpu", f"Expected 'cpu', got '{result}'"
        print("✅ Correctly falls back to CPU when CUDA unavailable")
    
    print()

def test_classifier_initialization():
    """Test that classifier can be initialized with mocked dependencies."""
    print("=== Test 3: Classifier initialization ===")
    
    # Mock all dependencies
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.device.return_value = "mocked_device"
    
    mock_config = MagicMock()
    mock_config.ml_model.classifier_confidence_threshold = 0.8
    mock_config.ml_model.use_gpu = True
    
    with patch.dict(sys.modules, {'torch': mock_torch}):
        with patch('bu_processor.bu_processor.pipeline.classifier.get_config', return_value=mock_config):
            with patch('bu_processor.bu_processor.core.config.get_config', return_value=mock_config):
                from bu_processor.bu_processor.pipeline.classifier import RealMLClassifier
                
                # Should not crash during initialization
                classifier = RealMLClassifier()
                print("✅ Classifier initialized successfully with mocked config")
                
                # Verify device was set
                assert hasattr(classifier, 'device'), "Classifier should have device attribute"
                print("✅ Device attribute properly set")
    
    print()

if __name__ == "__main__":
    print("Testing classifier device selection functionality...\n")
    
    try:
        test_device_selection_no_torch()
        test_device_selection_with_config()
        test_classifier_initialization()
        
        print("🎉 All device selection tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
