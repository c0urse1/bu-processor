from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Dict, Any

Role = Literal["user", "assistant", "system"]

@dataclass
class ChatTurn:
    role: Role
    content: str

@dataclass
class QueryPlan:
    """
    The output of the query understanding stage.
    """
    focused_query: str
    expanded_queries: List[str] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_queries(self) -> List[str]:
        """
        Focused first, then unique expansions (order preserved).
        """
        seen = set()
        out: List[str] = []
        for q in [self.focused_query, *self.expanded_queries]:
            qn = q.strip()
            if qn and qn not in seen:
                out.append(qn); seen.add(qn)
        return out
