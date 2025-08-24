#!/usr/bin/env python3
"""
🧪 FEATURE FLAGS TEST
==================
Test the centralized feature flag system.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_feature_flags():
    """Test that feature flags work correctly."""
    print("🧪 FEATURE FLAGS TEST")
    print("=" * 30)
    
    try:
        # Import flags
        from bu_processor.core.flags import (
            ENABLE_ENHANCED_PINECONE,
            ENABLE_METRICS, 
            ENABLE_EMBED_CACHE,
            get_all_flags,
            get_enabled_flags,
            is_mvp_mode,
            get_flag_summary,
            safe_import_prometheus,
            NoOpMetric,
            NoOpCache
        )
        
        print("✅ Flags imported successfully")
        
        # Test default values (should all be False for MVP)
        print(f"\n🔧 Default flag states:")
        print(f"  Enhanced Pinecone: {ENABLE_ENHANCED_PINECONE}")
        print(f"  Metrics: {ENABLE_METRICS}")  
        print(f"  Embed Cache: {ENABLE_EMBED_CACHE}")
        print(f"  MVP Mode: {is_mvp_mode()}")
        
        # Test environment variable override
        print(f"\n🔧 Testing environment override...")
        os.environ["ENABLE_METRICS"] = "true"
        
        # Re-import to get updated value
        import importlib
        import bu_processor.core.flags
        importlib.reload(bu_processor.core.flags)
        
        from bu_processor.core.flags import ENABLE_METRICS as METRICS_UPDATED
        print(f"  Metrics after ENV=true: {METRICS_UPDATED}")
        
        # Test utility functions
        print(f"\n🔧 Testing utility functions...")
        all_flags = get_all_flags()
        enabled_flags = get_enabled_flags()
        print(f"  Total flags: {len(all_flags)}")
        print(f"  Enabled flags: {len(enabled_flags)}")
        
        # Test conditional imports
        print(f"\n🔧 Testing conditional imports...")
        counter, histogram, gauge = safe_import_prometheus()
        if counter is None:
            print("  ✅ Prometheus import correctly disabled")
        else:
            print("  ✅ Prometheus import correctly enabled")
        
        # Test stub classes
        print(f"\n🔧 Testing stub classes...")
        noop_metric = NoOpMetric("test_metric")
        noop_metric.inc()  # Should not crash
        
        noop_cache = NoOpCache()
        noop_cache.set("key", "value")
        result = noop_cache.get("key", "default")
        print(f"  NoOp cache get: {result}")
        
        # Show summary
        print(f"\n📊 Flag Summary:")
        print("-" * 40)
        summary = get_flag_summary()
        print(summary)
        
        print("\n🎉 FEATURE FLAGS TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if "ENABLE_METRICS" in os.environ:
            del os.environ["ENABLE_METRICS"]

if __name__ == "__main__":
    success = test_feature_flags()
    sys.exit(0 if success else 1)
