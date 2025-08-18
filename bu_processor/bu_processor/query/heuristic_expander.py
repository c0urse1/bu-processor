from __future__ import annotations
from typing import List
import re

_SYNONYMS = {
    "insurance": ["coverage", "policy"],
    "financial": ["monetary", "economic"],
    "loss": ["damage", "liability"],
    "company": ["corporate", "business"],
    "pension": ["retirement", "annuity"],
    "error": ["mistake", "negligence"],
    "document": ["file", "pdf"],
}

_PATTERNS = [
    "explain {q}",
    "overview of {q}",
    "how does {q} work",
    "best practices for {q}",
]

class HeuristicExpander:
    """
    Deterministic, no-network:
    - Template paraphrases
    - Single-word synonym swap (at most one) to create variety
    """
    def expand(self, focused_query: str, num: int = 2) -> List[str]:
        q = re.sub(r"\s+", " ", focused_query.strip())
        out: List[str] = []
        
        # Add template-based expansions (limit to avoid dominating)
        for p in _PATTERNS[:2]:  # Take only first 2 templates
            out.append(p.format(q=q))

        # Add synonym-based expansions
        toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\-]+", q.lower())
        for i, t in enumerate(toks):
            if t in _SYNONYMS:
                for s in _SYNONYMS[t][:2]:
                    tt = toks.copy(); tt[i] = s
                    out.append(" ".join(tt))
                    if len(out) >= max(4, num * 2):  # Allow more expansions to be generated
                        break
            if len(out) >= max(4, num * 2):
                break

        seen, uniq = set(), []
        for x in out:
            x = re.sub(r"\s+", " ", x).strip()
            if x and x not in seen:
                uniq.append(x); seen.add(x)
        return uniq[: max(2, num)]
