#!/usr/bin/env python3
"""
🧪 FEATURE FLAGS DEMO
==================
Demonstrates how to use the centralized feature flag system.
"""

import os
import sys
from pathlib import Path

# Add project root to path  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def demo_feature_flags():
    """Demonstrate feature flag usage."""
    print("🧪 FEATURE FLAGS DEMO")
    print("=" * 40)
    
    # Import the flags
    from bu_processor.core.flags import (
        ENABLE_ENHANCED_PINECONE,
        ENABLE_METRICS,
        ENABLE_EMBED_CACHE,
        ENABLE_RERANK,
        get_all_flags,
        get_enabled_flags,
        is_mvp_mode,
        get_flag_summary,
        safe_import_prometheus,
        NoOpMetric,
        NoOpCache
    )
    
    print("📊 Current Flag State:")
    print("-" * 25)
    print(f"Enhanced Pinecone: {ENABLE_ENHANCED_PINECONE}")
    print(f"Metrics: {ENABLE_METRICS}")
    print(f"Embed Cache: {ENABLE_EMBED_CACHE}")
    print(f"Rerank: {ENABLE_RERANK}")
    print(f"MVP Mode: {is_mvp_mode()}")
    
    print(f"\n📈 Flag Statistics:")
    all_flags = get_all_flags()
    enabled_flags = get_enabled_flags()
    print(f"Total flags: {len(all_flags)}")
    print(f"Enabled flags: {len(enabled_flags)}")
    
    if enabled_flags:
        print("Enabled features:")
        for flag in enabled_flags.keys():
            print(f"  ✅ {flag}")
    else:
        print("  🎯 Running in MVP mode (no advanced features)")
    
    print(f"\n🔧 Conditional Import Example:")
    # Example: Conditionally use metrics
    counter_class, histogram_class, gauge_class = safe_import_prometheus()
    
    if counter_class is not None:
        print("  ✅ Prometheus metrics available")
        counter = counter_class("demo_counter", "Demo counter")
        counter.inc()
        print("  📊 Counter incremented")
    else:
        print("  🎯 Using no-op metrics for MVP")
        counter = NoOpMetric("demo_counter", "Demo counter")
        counter.inc()  # Does nothing
        print("  📊 No-op counter 'incremented'")
    
    print(f"\n🔧 Cache Example:")
    if ENABLE_EMBED_CACHE:
        print("  ✅ Embedding cache enabled")
        # Would use real cache here
    else:
        print("  🎯 Using no-op cache for MVP")
        cache = NoOpCache()
        cache.set("key", "value")
        result = cache.get("key", "default")
        print(f"  💾 Cache result: {result}")
    
    print(f"\n🔧 Feature-Conditional Code Example:")
    
    def process_with_features(text):
        """Example function that uses different features based on flags."""
        result = {"text": text}
        
        # Basic processing (always enabled)
        result["processed"] = text.lower()
        
        # Enhanced features (conditional)
        if ENABLE_ENHANCED_PINECONE:
            result["enhanced_vectors"] = True
            print("    🚀 Using enhanced Pinecone features")
        
        if ENABLE_RERANK:
            result["reranked"] = True
            print("    🎯 Applied reranking")
        
        if ENABLE_EMBED_CACHE:
            result["cached"] = True
            print("    💾 Used embedding cache")
        
        return result
    
    result = process_with_features("Sample text for processing")
    print(f"  📤 Processing result: {result}")
    
    print(f"\n📋 Full Summary:")
    print(get_flag_summary())
    
    print("\n🎉 FEATURE FLAGS DEMO COMPLETED!")

def demo_environment_override():
    """Demonstrate how to override flags with environment variables."""
    print("\n🧪 ENVIRONMENT OVERRIDE DEMO")
    print("=" * 45)
    
    # Set some flags via environment
    print("Setting environment variables:")
    os.environ["ENABLE_METRICS"] = "true"
    os.environ["ENABLE_RERANK"] = "true"
    print("  ENABLE_METRICS=true")
    print("  ENABLE_RERANK=true")
    
    # Re-import to see changes (in real usage, restart application)
    print("\nReloading flags...")
    import importlib
    import bu_processor.core.flags
    importlib.reload(bu_processor.core.flags)
    
    from bu_processor.core.flags import ENABLE_METRICS, ENABLE_RERANK, is_mvp_mode
    
    print(f"Updated flags:")
    print(f"  Metrics: {ENABLE_METRICS}")
    print(f"  Rerank: {ENABLE_RERANK}")
    print(f"  Still MVP mode: {is_mvp_mode()}")
    
    # Cleanup
    del os.environ["ENABLE_METRICS"]
    del os.environ["ENABLE_RERANK"]
    print("\nEnvironment cleaned up")

if __name__ == "__main__":
    try:
        demo_feature_flags()
        demo_environment_override()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
