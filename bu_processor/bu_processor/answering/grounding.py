from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
from statistics import mean
from bu_processor.retrieval.models import RetrievalHit

def score_confidence(hits: List[RetrievalHit]) -> float:
    if not hits:
        return 0.0
    # Normalize scores roughly to [0,1] if needed; here assume they are cosine/IP or BM25-like but we just scale
    scores = [max(0.0, min(1.0, float(h.score))) for h in hits[:5]]
    return mean(scores)

_NUM = re.compile(r'(?<![\w/])(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d+)?)(?![\w/])')

def detect_numeric_conflicts(hits: List[RetrievalHit], query: str, tolerance_ratio: float = 0.15) -> Tuple[bool, Dict[str, Any]]:
    """
    Heuristic: if top sources contain significantly different numeric claims, flag conflict.
    - Extract numbers from each top chunk; compare min/max.
    - If spread is big and numbers seem about the same quantity (same order of magnitude), flag it.
    """
    nums = []
    for h in hits[:5]:
        text = (h.metadata.get("summary") or h.text)
        vals = []
        for m in _NUM.findall(text):
            s = m.replace(",", "").replace(" ", "")
            try:
                vals.append(float(s))
            except Exception:
                pass
        if vals:
            nums.append(vals)
    flat = [v for lst in nums for v in lst]
    if len(flat) < 2:
        return False, {"reason": "not_enough_numbers"}
    mx, mn = max(flat), min(flat)
    if mn <= 0:
        return False, {"reason": "non_positive_or_sparse"}
    spread = (mx - mn) / max(mn, 1e-8)
    return (spread > tolerance_ratio), {"reason": "numeric_spread", "mn": mn, "mx": mx, "spread": spread}

def check_grounding_quality(
    hits: List[RetrievalHit],
    query: str,
    min_confidence: float = 0.3,
    min_sources: int = 2,
    max_conflicting_signals: int = 1
) -> Tuple[bool, str, List[str]]:
    """
    Assess whether retrieved content provides sufficient grounding for answering the query.
    
    Args:
        hits: Retrieved content chunks
        query: Original user query
        min_confidence: Minimum average confidence score
        min_sources: Minimum number of sources required
        max_conflicting_signals: Maximum number of conflicting indicators allowed
        
    Returns:
        (is_sufficient, reason, warnings)
    """
    if not hits:
        return False, "No sources found", []
    
    warnings = []
    
    # Check 1: Minimum confidence threshold
    avg_score = sum(h.score for h in hits) / len(hits)
    if avg_score < min_confidence:
        return False, f"Low confidence in sources (avg: {avg_score:.2f} < {min_confidence})", warnings
    
    # Check 2: Minimum number of sources
    unique_docs = set(h.metadata.get("doc_id", h.id) for h in hits)
    if len(unique_docs) < min_sources:
        return False, f"Insufficient source diversity ({len(unique_docs)} docs < {min_sources})", warnings
    
    # Check 3: Look for conflicting information signals
    query_terms = set(re.findall(r'\w+', query.lower()))
    conflicting_signals = 0
    
    # Simple heuristic: look for negation patterns in different sources
    negation_patterns = [r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bunlikely\b', r'\bfalse\b']
    sources_with_negation = []
    sources_without_negation = []
    
    for h in hits:
        text_lower = h.text.lower()
        has_negation = any(re.search(pattern, text_lower) for pattern in negation_patterns)
        if has_negation:
            sources_with_negation.append(h.metadata.get("doc_id", h.id))
        else:
            sources_without_negation.append(h.metadata.get("doc_id", h.id))
    
    # If we have both negation and non-negation sources, that might indicate conflict
    if sources_with_negation and sources_without_negation:
        conflicting_signals += 1
        warnings.append(f"Potential conflicting information detected across sources")
    
    # Check 4: Look for uncertainty language
    uncertainty_patterns = [r'\bmight\b', r'\bmay\b', r'\bpossibly\b', r'\buncertain\b', r'\bunknown\b']
    uncertain_sources = 0
    for h in hits:
        text_lower = h.text.lower()
        if any(re.search(pattern, text_lower) for pattern in uncertainty_patterns):
            uncertain_sources += 1
    
    if uncertain_sources > len(hits) * 0.6:  # More than 60% of sources express uncertainty
        warnings.append(f"High uncertainty detected in {uncertain_sources}/{len(hits)} sources")
    
    # Final decision
    if conflicting_signals > max_conflicting_signals:
        return False, f"Too many conflicting signals ({conflicting_signals} > {max_conflicting_signals})", warnings
    
    return True, "Sufficient grounding found", warnings


def format_insufficient_evidence_response(
    query: str, 
    reason: str, 
    warnings: List[str], 
    sources_table: List[Dict[str, Any]]
) -> str:
    """
    Format a response when grounding is insufficient.
    """
    response_parts = [
        f"I cannot provide a confident answer to '{query}' based on the available sources.",
        f"\nReason: {reason}"
    ]
    
    if warnings:
        response_parts.append(f"\nAdditional concerns:")
        for warning in warnings:
            response_parts.append(f"• {warning}")
    
    if sources_table:
        response_parts.append(f"\nSources consulted:")
        for i, source in enumerate(sources_table, 1):
            doc_info = source.get('doc_id', 'Unknown')
            if source.get('section'):
                doc_info += f" (section: {source['section']})"
            if source.get('page') is not None:
                doc_info += f" (page: {source['page']})"
            response_parts.append(f"[{i}] {doc_info}")
    
    response_parts.append(f"\nFor a more accurate answer, please provide additional context or try rephrasing your question.")
    
    return "\n".join(response_parts)
