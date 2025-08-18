# bu_processor/retrieval/filters.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Optional
from datetime import datetime

def _parse_date(s: str) -> datetime:
    # Expect 'YYYY-MM-DD' (extend if you store timestamps)
    return datetime.fromisoformat(s.strip())

def metadata_match(md: Dict[str, Any], flt: Optional[Dict[str, Any]]) -> bool:
    """
    Supports:
      - equality: {"section": "Finance"}
      - IN: {"section__in": ["Finance", "IR"]}
      - date range on field "date" (ISO): {"date_gte": "2024-01-01", "date_lte": "2025-12-31"}
    """
    if not flt:
        return True

    for k, v in flt.items():
        if k.endswith("__in"):
            key = k[:-4]
            vals: Iterable[Any] = v if isinstance(v, (list, tuple, set)) else [v]
            if md.get(key) not in vals:
                return False
        elif k in ("date_gte", "date_lte"):
            if "date" not in md or not md["date"]:
                return False
            d = _parse_date(str(md["date"]))
            if k == "date_gte" and d < _parse_date(str(v)):
                return False
            if k == "date_lte" and d > _parse_date(str(v)):
                return False
        else:
            if md.get(k) != v:
                return False
    return True
