from __future__ import annotations
from typing import Dict

def passes(aggregate: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    for k, th in thresholds.items():
        if aggregate.get(k, 0.0) < th:
            return False
    return True

def explain(aggregate: Dict[str, float], thresholds: Dict[str, float]) -> str:
    lines = ["Quality Gate:"]
    for k, th in thresholds.items():
        lines.append(f"- {k}: {aggregate.get(k, 0.0):.3f} (threshold {th:.3f})")
    return "\n".join(lines)
