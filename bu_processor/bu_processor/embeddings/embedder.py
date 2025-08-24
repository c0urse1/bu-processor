# bu_processor/embeddings/embedder.py
from __future__ import annotations
from typing import List, Dict, Optional, Any
import os
from ..core.flags import ENABLE_EMBED_CACHE

class Embedder:
    """
    Sentence transformer-based embedder with optional caching.
    
    Caching is controlled by ENABLE_EMBED_CACHE flag:
    - When True: Uses in-memory cache for embeddings
    - When False: No caching, direct computation every time
    
    The cache can be extended later with RLock, FIFO eviction, etc.
    when the flag is enabled.
    """
    
    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # 768-D
        )
        
        # Handle device specification
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        self.model = SentenceTransformer(self.model_name, device=device)
        
        # Cache setup based on flag
        self._cache: Optional[Dict[str, List[float]]] = {} if ENABLE_EMBED_CACHE else None
        
        # Dimension for checks/index creation
        try:
            self._dim = int(self.model.get_sentence_embedding_dimension())
        except Exception:
            self._dim = None

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dim:
            return self._dim
        # Fallback: determine once
        v = self.encode_one("test")
        self._dim = len(v)
        return self._dim
    
    @property
    def cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._cache is not None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._cache is None:
            return {"enabled": False, "size": 0}
        return {
            "enabled": True,
            "size": len(self._cache),
            "flag": ENABLE_EMBED_CACHE
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._cache is not None:
            self._cache.clear()

    def encode(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Encode multiple texts to embeddings.
        
        Uses caching if ENABLE_EMBED_CACHE=True, otherwise computes directly.
        """
        # Import metrics (No-Op if not enabled)
        try:
            from ..observability.metrics import (
                embedding_latency, embedding_cache_hits, embedding_cache_misses, time_operation
            )
            metrics_available = True
        except ImportError:
            metrics_available = False
        
        # Track operation timing
        timing_context = time_operation(embedding_latency, model=self.model_name, batch_size=str(batch_size)) if metrics_available else None
        
        if timing_context:
            timing_context.__enter__()
        
        try:
            if self._cache is None:
                # No caching - direct computation
                if metrics_available:
                    # All texts are cache misses when caching disabled
                    embedding_cache_misses.labels(cache_type="disabled").inc(len(texts))
                
                result = [v.tolist() for v in self.model.encode(texts, batch_size=batch_size, normalize_embeddings=False)]
                return result
            
            # Caching enabled - check cache and compute missing
            out = []
            not_cached = []
            idx = []
            cache_hits = 0
            
            # Check cache for each text
            for i, text in enumerate(texts):
                if text in self._cache:
                    out.append(self._cache[text])
                    cache_hits += 1
                else:
                    out.append(None)
                    not_cached.append(text)
                    idx.append(i)
            
            # Track cache hits/misses
            if metrics_available:
                if cache_hits > 0:
                    embedding_cache_hits.labels(cache_type="memory").inc(cache_hits)
                if len(not_cached) > 0:
                    embedding_cache_misses.labels(cache_type="memory").inc(len(not_cached))
            
            # Compute embeddings for uncached texts
            if not_cached:
                vecs = [v.tolist() for v in self.model.encode(not_cached, batch_size=batch_size, normalize_embeddings=False)]
                # Update cache and output
                for j, v in zip(idx, vecs):
                    out[j] = v
                    self._cache[texts[j]] = v
            
            return out
            
        finally:
            if timing_context:
                timing_context.__exit__(None, None, None)

    def encode_one(self, text: str) -> List[float]:
        """Encode single text to embedding."""
        return self.encode([text])[0]
    
    # Advanced cache methods (for future RLock/FIFO implementation)
    def _should_evict_cache(self) -> bool:
        """Determine if cache should be evicted (placeholder for future implementation)."""
        if self._cache is None:
            return False
        # Future: implement FIFO/LRU/size-based eviction
        return len(self._cache) > 10000  # Simple size limit for now
    
    def _evict_cache_entries(self) -> None:
        """Evict cache entries (placeholder for future implementation)."""
        if self._cache is None:
            return
        # Future: implement sophisticated eviction strategy
        # For now: simple clear when too large
        if self._should_evict_cache():
            self._cache.clear()
    
    def encode_with_eviction(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Encode texts with cache eviction check.
        
        This method can be enhanced later with RLock and FIFO eviction
        when the cache flag is enabled.
        """
        if self._cache is not None:
            self._evict_cache_entries()
        return self.encode(texts, batch_size)
