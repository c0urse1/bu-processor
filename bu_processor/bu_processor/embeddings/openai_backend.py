from __future__ import annotations
from typing import List
import numpy as np
from openai import OpenAI

class OpenAIEmbeddings:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        # Single call for many inputs; OpenAI handles batching internally
        resp = self.client.embeddings.create(model=self.model, input=texts)
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # Normalize for cosine
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.clip(norms, 1e-8, None)
        return arr
