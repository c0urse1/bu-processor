"""
Feature Flags for BU-Processor
==============================

Centralized feature flag system to control various components.
Nothing gets deleted - just deactivated until consciously enabled.
Production-like flags can be easily turned on later.
"""

import os
from typing import Dict, Any

def _on(name: str, default: bool = False) -> bool:
    """
    Check if a feature flag is enabled via environment variable.
    
    Args:
        name: Environment variable name
        default: Default value if env var is not set
        
    Returns:
        True if feature is enabled
    """
    val = os.getenv(name, str(default)).lower()
    return val in ("1", "true", "yes", "on")

# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Core Pinecone Features
ENABLE_ENHANCED_PINECONE = _on("ENABLE_ENHANCED_PINECONE", False)  # vollen Funktionsumfang
ENABLE_ASYNC_UPSERT      = _on("ENABLE_ASYNC_UPSERT", False)       # Batch-Async
ENABLE_STUB_MODE         = _on("ENABLE_PINECONE_STUB", False)      # „Trockenlauf"

# Performance & Caching
ENABLE_EMBED_CACHE       = _on("ENABLE_EMBED_CACHE", False)        # Embedding-Cache
ENABLE_THREADPOOL        = _on("ENABLE_THREADPOOL", False)         # ThreadPoolExecutor
ENABLE_RATE_LIMITER      = _on("ENABLE_RATE_LIMITER", False)       # Rate limiting

# Monitoring & Observability
ENABLE_METRICS           = _on("ENABLE_METRICS", False)            # Prometheus
ENABLE_DETAILED_LOGGING  = _on("ENABLE_DETAILED_LOGGING", False)   # Verbose logs
ENABLE_TRACING           = _on("ENABLE_TRACING", False)            # Distributed tracing

# ML & AI Features
ENABLE_RERANK            = _on("ENABLE_RERANK", False)             # Cross-Encoder Rerank
ENABLE_HYBRID_SEARCH     = _on("ENABLE_HYBRID_SEARCH", False)      # Sparse + Dense
ENABLE_QUERY_EXPANSION   = _on("ENABLE_QUERY_EXPANSION", False)    # Query enhancement

# Advanced Pipeline Features
ENABLE_CHUNK_OVERLAP     = _on("ENABLE_CHUNK_OVERLAP", False)      # Sliding window chunking
ENABLE_METADATA_EXTRACTION = _on("ENABLE_METADATA_EXTRACTION", False)  # Rich metadata
ENABLE_DEDUPLICATION     = _on("ENABLE_DEDUPLICATION", False)      # Content dedup

# API & Integration Features
ENABLE_CHATBOT           = _on("ENABLE_CHATBOT", False)            # Chatbot interface
ENABLE_STREAMING_API     = _on("ENABLE_STREAMING_API", False)      # Server-sent events
ENABLE_WEBHOOK_CALLBACKS = _on("ENABLE_WEBHOOK_CALLBACKS", False)  # Async callbacks

# Development & Testing
ENABLE_DEBUG_MODE        = _on("ENABLE_DEBUG_MODE", False)         # Debug features
ENABLE_MOCK_RESPONSES    = _on("ENABLE_MOCK_RESPONSES", False)     # Mock external APIs
ENABLE_PROFILING         = _on("ENABLE_PROFILING", False)          # Performance profiling

# Experimental Features
ENABLE_EXPERIMENTAL_CHUNKING = _on("ENABLE_EXPERIMENTAL_CHUNKING", False)  # New chunking algorithms
ENABLE_VECTOR_QUANTIZATION   = _on("ENABLE_VECTOR_QUANTIZATION", False)    # Compressed vectors
ENABLE_FEDERATED_SEARCH      = _on("ENABLE_FEDERATED_SEARCH", False)       # Multi-index search

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_all_flags() -> Dict[str, bool]:
    """Get all feature flags as a dictionary."""
    return {
        name: value for name, value in globals().items()
        if name.startswith("ENABLE_") and isinstance(value, bool)
    }

def get_enabled_flags() -> Dict[str, bool]:
    """Get only the enabled feature flags."""
    return {
        name: value for name, value in get_all_flags().items()
        if value is True
    }

def is_mvp_mode() -> bool:
    """Check if we're running in MVP mode (minimal features)."""
    enabled = get_enabled_flags()
    # MVP mode if no advanced features are enabled
    mvp_incompatible = [
        "ENABLE_ENHANCED_PINECONE",
        "ENABLE_ASYNC_UPSERT", 
        "ENABLE_EMBED_CACHE",
        "ENABLE_METRICS",
        "ENABLE_RERANK",
        "ENABLE_HYBRID_SEARCH"
    ]
    return not any(enabled.get(flag, False) for flag in mvp_incompatible)

def get_flag_summary() -> str:
    """Get a formatted summary of all feature flags."""
    all_flags = get_all_flags()
    enabled = sum(all_flags.values())
    total = len(all_flags)
    
    summary = [
        f"Feature Flags Summary: {enabled}/{total} enabled",
        f"MVP Mode: {'Yes' if is_mvp_mode() else 'No'}",
        ""
    ]
    
    # Group flags by category
    categories = {
        "Core": ["ENHANCED_PINECONE", "ASYNC_UPSERT", "STUB_MODE"],
        "Performance": ["EMBED_CACHE", "THREADPOOL", "RATE_LIMITER"],
        "Monitoring": ["METRICS", "DETAILED_LOGGING", "TRACING"],
        "ML/AI": ["RERANK", "HYBRID_SEARCH", "QUERY_EXPANSION"],
        "Pipeline": ["CHUNK_OVERLAP", "METADATA_EXTRACTION", "DEDUPLICATION"],
        "API": ["CHATBOT", "STREAMING_API", "WEBHOOK_CALLBACKS"],
        "Development": ["DEBUG_MODE", "MOCK_RESPONSES", "PROFILING"],
        "Experimental": ["EXPERIMENTAL_CHUNKING", "VECTOR_QUANTIZATION", "FEDERATED_SEARCH"]
    }
    
    for category, flag_suffixes in categories.items():
        summary.append(f"{category}:")
        for suffix in flag_suffixes:
            flag_name = f"ENABLE_{suffix}"
            status = "✅" if all_flags.get(flag_name, False) else "❌"
            summary.append(f"  {status} {flag_name}")
        summary.append("")
    
    return "\n".join(summary)

# ============================================================================
# CONDITIONAL IMPORTS (Examples)
# ============================================================================

def safe_import_prometheus():
    """Conditionally import Prometheus metrics."""
    if ENABLE_METRICS:
        try:
            from prometheus_client import Counter, Histogram, Gauge
            return Counter, Histogram, Gauge
        except ImportError:
            return None, None, None
    return None, None, None

def safe_import_async():
    """Conditionally import async libraries."""
    if ENABLE_ASYNC_UPSERT:
        try:
            import asyncio
            import aiohttp
            return asyncio, aiohttp
        except ImportError:
            return None, None
    return None, None

# ============================================================================
# STUB CLASSES FOR DISABLED FEATURES
# ============================================================================

class NoOpMetric:
    """No-operation metric for when metrics are disabled."""
    def __init__(self, *args, **kwargs):
        pass
    def inc(self, *args, **kwargs):
        pass
    def observe(self, *args, **kwargs):
        pass
    def set(self, *args, **kwargs):
        pass

class NoOpCache:
    """No-operation cache for when caching is disabled."""
    def __init__(self, *args, **kwargs):
        pass
    def get(self, key, default=None):
        return default
    def set(self, key, value, ttl=None):
        pass
    def clear(self):
        pass

# Default exports for easy importing
__all__ = [
    # Feature flags
    "ENABLE_ENHANCED_PINECONE",
    "ENABLE_ASYNC_UPSERT", 
    "ENABLE_EMBED_CACHE",
    "ENABLE_METRICS",
    "ENABLE_RERANK",
    "ENABLE_STUB_MODE",
    "ENABLE_THREADPOOL",
    "ENABLE_RATE_LIMITER",
    
    # Utility functions
    "get_all_flags",
    "get_enabled_flags", 
    "is_mvp_mode",
    "get_flag_summary",
    "safe_import_prometheus",
    "safe_import_async",
    
    # Stub classes
    "NoOpMetric",
    "NoOpCache",
    
    # Feature flags class
    "FeatureFlags"
]


class FeatureFlags:
    """
    Centralized feature flags access.
    
    Provides attribute-style access to all feature flags.
    """
    
    # Core Pinecone Features
    @property
    def enable_enhanced_pinecone(self) -> bool:
        return ENABLE_ENHANCED_PINECONE
    
    @property 
    def enable_async_upsert(self) -> bool:
        return ENABLE_ASYNC_UPSERT
    
    @property
    def enable_stub_mode(self) -> bool:
        return ENABLE_STUB_MODE
    
    # Performance & Caching
    @property
    def enable_embed_cache(self) -> bool:
        return ENABLE_EMBED_CACHE
    
    @property
    def enable_threadpool(self) -> bool:
        return ENABLE_THREADPOOL
    
    @property
    def enable_rate_limiter(self) -> bool:
        return ENABLE_RATE_LIMITER
    
    # Monitoring & Observability
    @property
    def enable_metrics(self) -> bool:
        return ENABLE_METRICS
    
    @property
    def enable_detailed_logging(self) -> bool:
        return ENABLE_DETAILED_LOGGING
    
    @property
    def enable_tracing(self) -> bool:
        return ENABLE_TRACING
    
    # ML & AI Features
    @property
    def enable_rerank(self) -> bool:
        return ENABLE_RERANK
    
    @property
    def enable_hybrid_search(self) -> bool:
        return ENABLE_HYBRID_SEARCH
    
    @property
    def enable_query_expansion(self) -> bool:
        return ENABLE_QUERY_EXPANSION
    
    @property
    def enable_debug_mode(self) -> bool:
        return ENABLE_DEBUG_MODE
    
    def get_all_flags(self) -> Dict[str, bool]:
        """Get all flags as a dictionary."""
        return {
            # Core Pinecone Features
            'enable_enhanced_pinecone': self.enable_enhanced_pinecone,
            'enable_async_upsert': self.enable_async_upsert,
            'enable_stub_mode': self.enable_stub_mode,
            
            # Performance & Caching
            'enable_embed_cache': self.enable_embed_cache,
            'enable_threadpool': self.enable_threadpool,
            'enable_rate_limiter': self.enable_rate_limiter,
            
            # Monitoring & Observability
            'enable_metrics': self.enable_metrics,
            'enable_detailed_logging': self.enable_detailed_logging,
            'enable_tracing': self.enable_tracing,
            
            # ML & AI Features
            'enable_rerank': self.enable_rerank,
            'enable_hybrid_search': self.enable_hybrid_search,
            'enable_query_expansion': self.enable_query_expansion,
            'enable_debug_mode': self.enable_debug_mode,
        }
    
    def get_enabled_flags(self) -> Dict[str, bool]:
        """Get only enabled flags."""
        return {k: v for k, v in self.get_all_flags().items() if v}
