#!/usr/bin/env python3
"""
Diagnostic test to understand device selection behavior.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def diagnose_device_selection():
    """Diagnose what's happening with device selection."""
    print("=== Device Selection Diagnosis ===\n")
    
    # Check torch availability
    try:
        import torch
        print(f"✅ torch imported successfully")
        print(f"   torch version: {torch.__version__}")
        print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device count: {torch.cuda.device_count()}")
            print(f"   Current CUDA device: {torch.cuda.current_device()}")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        torch = None
    except Exception as e:
        print(f"❌ torch check failed: {e}")
        torch = None
    
    print()
    
    # Test our device selection function
    from bu_processor.bu_processor.pipeline.classifier import _pick_device
    
    print("Testing _pick_device function:")
    print(f"   _pick_device(True): {_pick_device(True)}")
    print(f"   _pick_device(False): {_pick_device(False)}")
    
    print()
    
    # Test config loading
    try:
        from bu_processor.bu_processor.core.config import get_config
        cfg = get_config()
        print(f"✅ Config loaded successfully")
        print(f"   cfg.ml_model.use_gpu: {cfg.ml_model.use_gpu}")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
    
    print()
    
    # Check environment variable
    print(f"Environment variables:")
    print(f"   BU_USE_GPU: {os.environ.get('BU_USE_GPU', 'not set')}")

if __name__ == "__main__":
    diagnose_device_selection()
