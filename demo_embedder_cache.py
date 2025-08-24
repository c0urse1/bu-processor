#!/usr/bin/env python3
"""
Demo: Embedder with Flag-Controlled Caching

This script demonstrates how the embedder caching works with the
ENABLE_EMBED_CACHE feature flag. When disabled, no caching occurs.
When enabled, embeddings are cached for faster repeated access.
"""
import os
import sys
import time
from pathlib import Path

# Add the bu_processor package to path
sys.path.insert(0, str(Path(__file__).parent / "bu_processor"))

def demo_embedder_without_cache():
    """Demonstrate embedder behavior with caching disabled."""
    print("=" * 60)
    print("EMBEDDER WITHOUT CACHE (ENABLE_EMBED_CACHE=False)")
    print("=" * 60)
    
    # Ensure cache is disabled
    os.environ.pop("ENABLE_EMBED_CACHE", None)
    
    # Import after setting environment
    from bu_processor.embeddings.embedder import Embedder
    
    print("\n1. Creating embedder with caching disabled...")
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Smaller model for demo
    
    print(f"   Model: {embedder.model_name}")
    print(f"   Dimension: {embedder.dimension}")
    print(f"   Cache enabled: {embedder.cache_enabled}")
    print(f"   Cache stats: {embedder.get_cache_stats()}")
    
    # Test texts
    texts = [
        "This is a test document",
        "Another test document", 
        "This is a test document",  # Repeat
        "Final test document"
    ]
    
    print(f"\n2. Encoding {len(texts)} texts (with repeat)...")
    start_time = time.time()
    
    embeddings = embedder.encode(texts)
    
    elapsed = time.time() - start_time
    print(f"   Encoding took: {elapsed:.3f} seconds")
    print(f"   Embedding dimensions: {[len(emb) for emb in embeddings]}")
    print(f"   Cache stats after: {embedder.get_cache_stats()}")
    
    # Test repeated encoding
    print("\n3. Encoding same texts again (no cache benefit)...")
    start_time = time.time()
    
    embeddings2 = embedder.encode(texts)
    
    elapsed2 = time.time() - start_time
    print(f"   Second encoding took: {elapsed2:.3f} seconds")
    print(f"   Same results: {embeddings == embeddings2}")
    print(f"   Cache stats after: {embedder.get_cache_stats()}")
    print(f"   Time savings: None (no caching)")


def demo_embedder_with_cache():
    """Demonstrate embedder behavior with caching enabled."""
    print("\n" + "=" * 60)
    print("EMBEDDER WITH CACHE (ENABLE_EMBED_CACHE=True)")
    print("=" * 60)
    
    # Enable cache
    os.environ["ENABLE_EMBED_CACHE"] = "true"
    
    # Need to reload module to pick up flag change
    # In production, this would be set at startup
    print("\n1. Creating embedder with caching enabled...")
    
    # Import fresh instance (in real app, flag would be set before import)
    import importlib
    if 'bu_processor.embeddings.embedder' in sys.modules:
        importlib.reload(sys.modules['bu_processor.embeddings.embedder'])
    
    from bu_processor.embeddings.embedder import Embedder
    
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"   Model: {embedder.model_name}")
    print(f"   Dimension: {embedder.dimension}")
    print(f"   Cache enabled: {embedder.cache_enabled}")
    print(f"   Cache stats: {embedder.get_cache_stats()}")
    
    # Test texts
    texts = [
        "This is a test document",
        "Another test document", 
        "This is a test document",  # Repeat
        "Final test document"
    ]
    
    print(f"\n2. Encoding {len(texts)} texts (with repeat)...")
    start_time = time.time()
    
    embeddings = embedder.encode(texts)
    
    elapsed = time.time() - start_time
    print(f"   Encoding took: {elapsed:.3f} seconds")
    print(f"   Embedding dimensions: {[len(emb) for emb in embeddings]}")
    print(f"   Cache stats after: {embedder.get_cache_stats()}")
    
    # Test repeated encoding
    print("\n3. Encoding same texts again (cache benefit)...")
    start_time = time.time()
    
    embeddings2 = embedder.encode(texts)
    
    elapsed2 = time.time() - start_time
    print(f"   Second encoding took: {elapsed2:.3f} seconds")
    print(f"   Same results: {embeddings == embeddings2}")
    print(f"   Cache stats after: {embedder.get_cache_stats()}")
    
    if elapsed > 0:
        speedup = elapsed / elapsed2 if elapsed2 > 0 else float('inf')
        print(f"   Speedup: {speedup:.1f}x faster")
    
    # Test individual encoding
    print("\n4. Testing single text encoding...")
    single_text = "Single test text"
    
    start_time = time.time()
    single_emb1 = embedder.encode_one(single_text)
    elapsed_single1 = time.time() - start_time
    
    start_time = time.time()
    single_emb2 = embedder.encode_one(single_text)  # Should be cached
    elapsed_single2 = time.time() - start_time
    
    print(f"   First encoding: {elapsed_single1:.3f}s")
    print(f"   Second encoding: {elapsed_single2:.3f}s (cached)")
    print(f"   Same result: {single_emb1 == single_emb2}")
    print(f"   Final cache stats: {embedder.get_cache_stats()}")


def demo_cache_management():
    """Demonstrate cache management features."""
    print("\n" + "=" * 60)
    print("CACHE MANAGEMENT FEATURES")
    print("=" * 60)
    
    # Ensure cache is enabled
    os.environ["ENABLE_EMBED_CACHE"] = "true"
    
    from bu_processor.embeddings.embedder import Embedder
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("\n1. Testing cache management...")
    
    # Add some items to cache
    texts = [f"Test text {i}" for i in range(5)]
    embedder.encode(texts)
    
    print(f"   Cache after encoding 5 texts: {embedder.get_cache_stats()}")
    
    # Clear cache
    print("\n2. Clearing cache...")
    embedder.clear_cache()
    print(f"   Cache after clearing: {embedder.get_cache_stats()}")
    
    # Test eviction method
    print("\n3. Testing cache with eviction check...")
    more_texts = [f"Additional text {i}" for i in range(3)]
    embedder.encode_with_eviction(more_texts)
    print(f"   Cache after eviction-aware encoding: {embedder.get_cache_stats()}")


def demo_integration_with_metrics():
    """Show how embedder integrates with metrics system."""
    print("\n" + "=" * 60)
    print("INTEGRATION WITH METRICS SYSTEM")
    print("=" * 60)
    
    # Enable both cache and metrics
    os.environ["ENABLE_EMBED_CACHE"] = "true"
    os.environ["ENABLE_METRICS"] = "false"  # Keep as No-Op for demo
    
    from bu_processor.embeddings.embedder import Embedder
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("\n1. Embedder with metrics integration (No-Op mode)...")
    
    # Test with cache hits and misses
    texts = ["Text A", "Text B", "Text A", "Text C", "Text B"]
    print(f"   Encoding texts: {texts}")
    
    embeddings = embedder.encode(texts)
    
    print(f"   Results: {len(embeddings)} embeddings")
    print(f"   Cache stats: {embedder.get_cache_stats()}")
    print("   Metrics: Cache hits/misses tracked (No-Op mode)")
    print("   Timing: Embedding latency tracked (No-Op mode)")
    
    print("\n   (With ENABLE_METRICS=true, would see real Prometheus metrics)")


def demo_flag_benefits():
    """Demonstrate the benefits of flag-controlled caching."""
    print("\n" + "=" * 60)
    print("FLAG-CONTROLLED CACHING BENEFITS")
    print("=" * 60)
    
    print("\n✅ CACHE REMAINS AVAILABLE:")
    print("   - All caching logic preserved in code")
    print("   - No conditional 'if' statements in business logic")
    print("   - Cache can be enabled instantly via environment variable")
    
    print("\n✅ MVP-READY:")
    print("   - Default ENABLE_EMBED_CACHE=false = No cache = Less memory")
    print("   - Simple operation for MVP without cache complexity")
    print("   - No cache management overhead when disabled")
    
    print("\n✅ PERFORMANCE TUNING:")
    print("   - Enable cache in production: ENABLE_EMBED_CACHE=true")
    print("   - Immediate performance boost for repeated embeddings")
    print("   - Can measure cache hit rates with metrics")
    
    print("\n✅ FUTURE EXTENSIBILITY:")
    print("   - Ready for RLock threading safety")
    print("   - Ready for FIFO/LRU eviction strategies")
    print("   - Ready for persistent cache backends")
    print("   - All controlled by same flag system")
    
    print("\n✅ MEMORY CONTROL:")
    print("   - No memory usage when cache disabled")
    print("   - Configurable eviction when enabled")
    print("   - Cache size monitoring through metrics")


if __name__ == "__main__":
    print("🚀 Embedder Caching Demo")
    print("=" * 60)
    print("Demonstrating flag-controlled embedding cache")
    
    try:
        demo_embedder_without_cache()
        demo_embedder_with_cache()
        demo_cache_management()
        demo_integration_with_metrics()
        demo_flag_benefits()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed!")
        print("\nTo enable caching in production:")
        print("  export ENABLE_EMBED_CACHE=true")
        print("To enable metrics:")
        print("  export ENABLE_METRICS=true")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n⚠️  Import error: {e}")
        print("This demo requires sentence-transformers.")
        print("Install with: pip install sentence-transformers")
        print("\nCaching logic is still available and flag-controlled!")
