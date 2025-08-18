# bu_processor/semantic/embeddings.py
from __future__ import annotations
from typing import List, Protocol, runtime_checkable, Optional
import numpy as np

@runtime_checkable
class EmbeddingsBackend(Protocol):
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        ...

class SbertEmbeddings:
    """
    Production embedding backend using Sentence-Transformers.
    Lazily loads the model to avoid import cost in non-semantic paths.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer  # local import
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        # returns float32 np.ndarray [N, D]
        vecs = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)
