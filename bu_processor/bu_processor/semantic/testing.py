# bu_processor/semantic/testing.py
from __future__ import annotations
from typing import List
import numpy as np
import hashlib

class FakeDeterministicEmbeddings:
    """
    Deterministic "semantic-ish" embedder:
    - creates a pseudo-vector seeded by stable SHA1 hash of the text
    - vectors are normalized; similar texts -> similar seeds -> similar vectors on average
    - No network; perfect for unit tests.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for t in texts:
            h = hashlib.sha1(t.strip().lower().encode("utf-8")).digest()  # 20 bytes
            seed = int.from_bytes(h[:8], "big", signed=False) & 0x7FFFFFFFFFFFFFFF
            rng = np.random.default_rng(seed)
            v = rng.normal(size=self.dim).astype(np.float32)
            # "keyword drift" to help tests separate topics deterministically
            for kw, idx in (("cat", 0), ("dog", 1), ("finance", 2), ("health", 3), ("insurance", 4)):
                if kw in t.lower():
                    v[idx] += 5.0
            # normalize
            v /= max(np.linalg.norm(v), 1e-8)
            out.append(v)
        return np.stack(out)
