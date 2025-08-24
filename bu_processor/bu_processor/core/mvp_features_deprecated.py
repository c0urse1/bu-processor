#!/usr/bin/env python3
"""
MVP Feature Management - DEPRECATED
==================================

This module is deprecated. Use bu_processor.core.flags instead.
Keeping for backward compatibility only.
"""

# Import from the new centralized flags module
try:
    from .flags import (
        ENABLE_METRICS,
        ENABLE_EMBED_CACHE as ENABLE_EMBEDDING_CACHE,
        ENABLE_THREADPOOL,
        NoOpMetric,
        NoOpCache,
        safe_import_prometheus
    )
    
    # Backward compatibility aliases
    VECTOR_DB_ENABLE = True  # Always true for MVP, controlled by config instead
    
    def safe_import_threadpool():
        """Safely import ThreadPoolExecutor only if enabled."""
        if ENABLE_THREADPOOL:
            try:
                from concurrent.futures import ThreadPoolExecutor
                return ThreadPoolExecutor
            except ImportError:
                return None
        return None
    
    def get_metrics_or_noop():
        """Get metrics or no-op stubs."""
        if ENABLE_METRICS:
            counter, histogram, gauge = safe_import_prometheus()
            if counter is not None:
                return counter, histogram, gauge
        return NoOpMetric, NoOpMetric, NoOpMetric
    
    # Legacy class for compatibility
    class MVPFeatureFlags:
        """Legacy MVP feature flags - use bu_processor.core.flags instead."""
        ENABLE_METRICS = ENABLE_METRICS
        ENABLE_EMBEDDING_CACHE = ENABLE_EMBEDDING_CACHE  
        ENABLE_THREADPOOL = ENABLE_THREADPOOL
        USE_THREAD_POOL_EXECUTOR = ENABLE_THREADPOOL  # Alias
        
        @classmethod
        def get_all_flags(cls):
            return {
                "ENABLE_METRICS": cls.ENABLE_METRICS,
                "ENABLE_EMBEDDING_CACHE": cls.ENABLE_EMBEDDING_CACHE,
                "ENABLE_THREADPOOL": cls.ENABLE_THREADPOOL
            }
        
        @classmethod 
        def print_status(cls):
            """Print current feature flag status."""
            print("🎯 MVP Feature Flags Status (DEPRECATED - use flags.py):")
            print("=" * 50)
            for flag, value in cls.get_all_flags().items():
                status = "✅" if value else "❌"
                print(f"  {status} {flag}: {value}")

except ImportError:
    # Fallback if flags module not available
    import os
    
    def _get_flag(name: str, default: bool = False) -> bool:
        val = os.getenv(name, str(default)).lower()
        return val in ("1", "true", "yes", "on")
    
    ENABLE_METRICS = _get_flag("ENABLE_METRICS", False)
    ENABLE_EMBEDDING_CACHE = _get_flag("ENABLE_EMBED_CACHE", False)
    ENABLE_THREADPOOL = _get_flag("ENABLE_THREADPOOL", False)
    VECTOR_DB_ENABLE = True
    
    class NoOpMetric:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
    
    def safe_import_threadpool():
        if ENABLE_THREADPOOL:
            try:
                from concurrent.futures import ThreadPoolExecutor
                return ThreadPoolExecutor
            except ImportError:
                return None
        return None
    
    def get_metrics_or_noop():
        return NoOpMetric, NoOpMetric, NoOpMetric
    
    class MVPFeatureFlags:
        ENABLE_METRICS = ENABLE_METRICS
        ENABLE_EMBEDDING_CACHE = ENABLE_EMBEDDING_CACHE
        ENABLE_THREADPOOL = ENABLE_THREADPOOL
        USE_THREAD_POOL_EXECUTOR = ENABLE_THREADPOOL
        
        @classmethod
        def get_all_flags(cls):
            return {
                "ENABLE_METRICS": cls.ENABLE_METRICS,
                "ENABLE_EMBEDDING_CACHE": cls.ENABLE_EMBEDDING_CACHE,
                "ENABLE_THREADPOOL": cls.ENABLE_THREADPOOL
            }
        
        @classmethod
        def print_status(cls):
            print("🎯 MVP Feature Flags Status (Fallback):")
            print("=" * 40)
            for flag, value in cls.get_all_flags().items():
                status = "✅" if value else "❌"
                print(f"  {status} {flag}: {value}")

# =============================================================================
# CACHE DECORATORS
# =============================================================================

def mvp_cache(func):
    """
    Cache decorator that respects MVP feature flags.
    For MVP, just executes the function without caching.
    """
    def wrapper(*args, **kwargs):
        # For MVP, always just execute without caching
        return func(*args, **kwargs)
    return wrapper

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MVPFeatureFlags",
    "get_metrics_or_noop", 
    "safe_import_threadpool",
    "NoOpMetric",
    "mvp_cache",
    "ENABLE_METRICS",
    "ENABLE_EMBEDDING_CACHE", 
    "ENABLE_THREADPOOL",
    "VECTOR_DB_ENABLE"
]
