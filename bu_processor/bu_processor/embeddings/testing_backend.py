from __future__ import annotations
from typing import List
import numpy as np, hashlib

class FakeDeterministicEmbeddings:
    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for t in texts:
            h = hashlib.sha1(t.strip().lower().encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "big") & 0x7FFFFFFFFFFFFFFF
            rng = np.random.default_rng(seed)
            v = rng.normal(size=self.dim).astype(np.float32)
            # nudge for a few keywords to make tests meaningful
            keys = ["insurance", "finance", "health", "cat", "dog"]
            for i, k in enumerate(keys[: min(5, self.dim)]):
                if k in t.lower():
                    v[i] += 6.0
            v /= max(np.linalg.norm(v), 1e-8)
            out.append(v)
        return np.stack(out)
