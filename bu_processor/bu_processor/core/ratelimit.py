# bu_processor/core/ratelimit.py
"""
Rate limiting functionality with feature flag control.

This module provides rate limiting decorators and context managers that can be
disabled via feature flags. When ENABLE_RATE_LIMITER=False, all rate limiting
becomes No-Op but preserves the API.

Usage:
    from bu_processor.core.ratelimit import rate_limited, RateLimiter
    
    @rate_limited(calls_per_second=10)
    def api_call():
        pass
    
    # Or with context manager
    limiter = RateLimiter(calls_per_second=5)
    with limiter:
        make_request()
"""
import time
import asyncio
from typing import Callable, Any, Optional, Union
from functools import wraps
from ..core.flags import ENABLE_RATE_LIMITER

# ============================================================================
# No-Op Rate Limiter Classes
# ============================================================================

class _NoOpRateLimiter:
    """No-Op rate limiter that does nothing but preserves the API."""
    
    def __init__(self, calls_per_second: float = 1.0, 
                 burst_size: Optional[int] = None,
                 name: str = "noop"):
        self.calls_per_second = calls_per_second
        self.burst_size = burst_size
        self.name = name
    
    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens (No-Op - always succeeds immediately)."""
        return True
    
    async def acquire_async(self, tokens: int = 1) -> bool:
        """Acquire tokens asynchronously (No-Op - always succeeds immediately)."""
        return True
    
    def __enter__(self):
        """Context manager entry (No-Op)."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (No-Op)."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry (No-Op)."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit (No-Op)."""
        pass


class _TokenBucketRateLimiter:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, calls_per_second: float = 1.0,
                 burst_size: Optional[int] = None,
                 name: str = "token_bucket"):
        self.rate = calls_per_second
        self.burst_size = burst_size or int(calls_per_second * 2)
        self.name = name
        
        self.tokens = float(self.burst_size)
        self.last_update = time.time()
        self._lock = asyncio.Lock() if asyncio._get_running_loop() else None
    
    def _update_tokens(self):
        """Update available tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
        self.last_update = now
    
    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens (blocking if needed)."""
        self._update_tokens()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        # Wait for tokens to become available
        wait_time = (tokens - self.tokens) / self.rate
        time.sleep(wait_time)
        self.tokens = max(0, self.tokens - tokens)
        return True
    
    async def acquire_async(self, tokens: int = 1) -> bool:
        """Acquire tokens asynchronously."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        
        async with self._lock:
            self._update_tokens()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            # Wait for tokens to become available
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = max(0, self.tokens - tokens)
            return True
    
    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire_async()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


# ============================================================================
# Conditional Rate Limiter Class
# ============================================================================

if ENABLE_RATE_LIMITER:
    RateLimiterClass = _TokenBucketRateLimiter
    _rate_limiting_enabled = True
else:
    RateLimiterClass = _NoOpRateLimiter
    _rate_limiting_enabled = False


def RateLimiter(calls_per_second: float = 1.0, 
               burst_size: Optional[int] = None,
               name: str = "default") -> Union[_TokenBucketRateLimiter, _NoOpRateLimiter]:
    """Create a rate limiter instance."""
    return RateLimiterClass(calls_per_second, burst_size, name)


# ============================================================================
# Decorator Functions
# ============================================================================

def rate_limited(calls_per_second: float = 1.0,
                burst_size: Optional[int] = None,
                name: Optional[str] = None):
    """
    Decorator to rate limit function calls.
    
    Args:
        calls_per_second: Maximum calls per second
        burst_size: Maximum burst size (defaults to 2x calls_per_second)
        name: Name for the rate limiter
    
    Usage:
        @rate_limited(calls_per_second=10)
        def api_call():
            pass
    """
    def decorator(func: Callable) -> Callable:
        if not ENABLE_RATE_LIMITER:
            # Return original function unchanged when rate limiting disabled
            return func
        
        limiter_name = name or f"{func.__name__}_limiter"
        limiter = RateLimiter(calls_per_second, burst_size, limiter_name)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.acquire()
            return func(*args, **kwargs)
        
        # Store limiter on function for inspection
        wrapper._rate_limiter = limiter
        return wrapper
    
    return decorator


def async_rate_limited(calls_per_second: float = 1.0,
                      burst_size: Optional[int] = None,
                      name: Optional[str] = None):
    """
    Decorator to rate limit async function calls.
    
    Args:
        calls_per_second: Maximum calls per second
        burst_size: Maximum burst size (defaults to 2x calls_per_second)
        name: Name for the rate limiter
    
    Usage:
        @async_rate_limited(calls_per_second=10)
        async def api_call():
            pass
    """
    def decorator(func: Callable) -> Callable:
        if not ENABLE_RATE_LIMITER:
            # Return original function unchanged when rate limiting disabled
            return func
        
        limiter_name = name or f"{func.__name__}_async_limiter"
        limiter = RateLimiter(calls_per_second, burst_size, limiter_name)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await limiter.acquire_async()
            return await func(*args, **kwargs)
        
        # Store limiter on function for inspection
        wrapper._rate_limiter = limiter
        return wrapper
    
    return decorator


# ============================================================================
# Pre-configured Rate Limiters
# ============================================================================

# Pinecone API rate limiter (conservative defaults)
pinecone_limiter = RateLimiter(
    calls_per_second=10.0,  # 10 calls per second
    burst_size=20,          # Allow bursts up to 20
    name="pinecone_api"
)

# Embedding API rate limiter
embedding_limiter = RateLimiter(
    calls_per_second=5.0,   # 5 calls per second
    burst_size=10,          # Allow bursts up to 10
    name="embedding_api"
)

# PDF processing rate limiter
pdf_processing_limiter = RateLimiter(
    calls_per_second=2.0,   # 2 PDFs per second
    burst_size=5,           # Allow bursts up to 5
    name="pdf_processing"
)

# General API rate limiter
api_limiter = RateLimiter(
    calls_per_second=20.0,  # 20 calls per second
    burst_size=50,          # Allow bursts up to 50
    name="general_api"
)


# ============================================================================
# Utility Functions
# ============================================================================

def is_rate_limiting_enabled() -> bool:
    """Check if rate limiting is enabled."""
    return _rate_limiting_enabled


def get_rate_limit_info() -> dict:
    """Get information about rate limiting system status."""
    return {
        "enabled": ENABLE_RATE_LIMITER,
        "implementation": "token_bucket" if _rate_limiting_enabled else "noop",
        "limiters": [
            {"name": "pinecone_api", "rate": 10.0, "burst": 20},
            {"name": "embedding_api", "rate": 5.0, "burst": 10},
            {"name": "pdf_processing", "rate": 2.0, "burst": 5},
            {"name": "general_api", "rate": 20.0, "burst": 50}
        ]
    }


# ============================================================================
# Context Manager Helpers
# ============================================================================

class RateLimitedOperation:
    """Context manager for rate-limited operations."""
    
    def __init__(self, limiter: Union[_TokenBucketRateLimiter, _NoOpRateLimiter],
                 operation_name: str = "operation"):
        self.limiter = limiter
        self.operation_name = operation_name
    
    def __enter__(self):
        self.limiter.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def __aenter__(self):
        await self.limiter.acquire_async()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def rate_limited_operation(limiter: Union[_TokenBucketRateLimiter, _NoOpRateLimiter],
                          operation_name: str = "operation") -> RateLimitedOperation:
    """Create a rate-limited operation context manager."""
    return RateLimitedOperation(limiter, operation_name)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main classes
    "RateLimiter",
    "RateLimiterClass",
    
    # Decorators
    "rate_limited",
    "async_rate_limited",
    
    # Pre-configured limiters
    "pinecone_limiter",
    "embedding_limiter", 
    "pdf_processing_limiter",
    "api_limiter",
    
    # Context managers
    "RateLimitedOperation",
    "rate_limited_operation",
    
    # Utilities
    "is_rate_limiting_enabled",
    "get_rate_limit_info",
    
    # Implementation classes (for direct use if needed)
    "_NoOpRateLimiter",
    "_TokenBucketRateLimiter"
]
