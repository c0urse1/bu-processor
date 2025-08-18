from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Citation:
    paragraph_idx: int
    chunk_id: str
    doc_id: str

@dataclass
class AnswerResult:
    text: str
    citations: List[Citation] = field(default_factory=list)
    sources_table: List[Dict[str, Any]] = field(default_factory=list)  # idx->source meta (1-based indices)
    trace: Dict[str, Any] = field(default_factory=dict)
