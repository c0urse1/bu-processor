from __future__ import annotations
from typing import List, Optional
import numpy as np

class SbertEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        v = self.model.encode(texts, batch_size=batch_size,
                              convert_to_numpy=True, normalize_embeddings=True)
        return v.astype(np.float32)
