from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

class PineconeIndex:
    """
    Pinecone v4+ style client.
    """
    def __init__(self, api_key: str, index_name: str, cloud: str = "aws", region: str = "us-east-1"):
        from pinecone import Pinecone, ServerlessSpec  # lazy import
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        if index_name not in [i.name for i in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=1,  # placeholder, reset in create()
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self.index = self.pc.Index(index_name)
        self._dim = None

    def create(self, dim: int, metric: str = "cosine", namespace: Optional[str] = None) -> None:
        # Pinecone index must exist with correct dimension; recreate if needed
        info = self.pc.describe_index(self.index_name)
        if info.dimension != dim or info.metric != metric:
            self.pc.delete_index(self.index_name)
            from pinecone import ServerlessSpec
            self.pc.create_index(
                name=self.index_name,
                dimension=dim,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            self.index = self.pc.Index(self.index_name)
        self._dim = dim

    def upsert(self, ids: List[str], vectors: np.ndarray, metadata: List[Dict[str, Any]],
               namespace: Optional[str] = None) -> None:
        if self._dim is None:
            raise RuntimeError("Call create(dim=...) first.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vecs = (vectors / np.clip(norms, 1e-8, None)).astype(np.float32)
        payload = [{"id": _id, "values": vecs[i].tolist(), "metadata": metadata[i]} for i, _id in enumerate(ids)]
        self.index.upsert(vectors=payload, namespace=namespace or "")

    def query(self, vector: np.ndarray, top_k: int = 10, namespace: Optional[str] = None,
              metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        v = vector.astype(np.float32)
        v /= max(np.linalg.norm(v), 1e-8)
        resp = self.index.query(vector=v.tolist(), top_k=top_k, include_metadata=True,
                                namespace=namespace or "", filter=metadata_filter)
        out = []
        for m in resp.matches or []:
            out.append({"id": m.id, "score": float(m.score), "metadata": dict(m.metadata or {})})
        return out

    def delete(self, ids: List[str], namespace: Optional[str] = None) -> None:
        self.index.delete(ids=ids, namespace=namespace or "")
