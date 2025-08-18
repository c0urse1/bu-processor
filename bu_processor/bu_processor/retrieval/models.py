# bu_processor/retrieval/models.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class RetrievalHit:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
