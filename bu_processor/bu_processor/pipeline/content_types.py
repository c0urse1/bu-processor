"""Unified content type enum used across pipeline modules.

This consolidates previously duplicated definitions that lived in:
- semantic_chunking_enhancement.py
- simhash_semantic_deduplication.py
- demo/test sections redefining a slim version

Adding new members should consider backward compatibility. Where prior
enums used slightly different names (e.g. TECHNICAL vs TECHNICAL_SPEC,
TABLE_HEAVY vs TABLE) we include all distinct values so existing persisted
data or serialized references keep working.
"""

from enum import Enum


class ContentType(Enum):
    """Canonical content type set (superset of all prior variants)."""
    LEGAL_TEXT = "legal_text"
    TECHNICAL = "technical"  # generic technical content (old semantic enhancer)
    TECHNICAL_SPEC = "technical_spec"  # specific technical specification (deduplicator)
    TABLE_HEAVY = "table_heavy"  # text with many tables
    TABLE = "table"  # isolated table chunks
    LIST = "list"  # list-heavy content
    NARRATIVE = "narrative"
    MIXED = "mixed"  # heterogeneous / fallback
    UNKNOWN = "unknown"  # safety fallback

    @classmethod
    def from_str(cls, value: str) -> "ContentType":  # pragma: no cover - simple helper
        """Robust conversion from arbitrary string; falls back to MIXED/UNKNOWN."""
        normalized = (value or "").strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        # Backwards-compatible simple heuristics
        if "table" in normalized:
            return cls.TABLE_HEAVY if "heavy" in normalized else cls.TABLE
        if "tech" in normalized:
            return cls.TECHNICAL_SPEC if "spec" in normalized else cls.TECHNICAL
        if "legal" in normalized:
            return cls.LEGAL_TEXT
        if "narr" in normalized:
            return cls.NARRATIVE
        if "list" in normalized:
            return cls.LIST
        return cls.MIXED

__all__ = ["ContentType"]
