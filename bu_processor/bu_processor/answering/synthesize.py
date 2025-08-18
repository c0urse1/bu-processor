from __future__ import annotations
from typing import List, Optional, Dict, Any
from bu_processor.retrieval.models import RetrievalHit
from bu_processor.answering.context_packer import pack_context
from bu_processor.answering.grounding import score_confidence, detect_numeric_conflicts
from bu_processor.answering.models import AnswerResult
from bu_processor.ports import LlmAnswerer

def synthesize_answer(
    *,
    query: str,
    hits: List[RetrievalHit],
    answerer: LlmAnswerer,
    token_budget: int = 1200,
    min_confidence: float = 0.25,
    allow_conflicts: bool = False,
) -> AnswerResult:
    """
    Orchestrate the complete answer synthesis process with grounding checks.
    
    Args:
        query: User's question
        hits: Retrieved and reranked content chunks
        answerer: LLM or rule-based answerer implementation
        token_budget: Maximum tokens for context packing
        min_confidence: Minimum confidence threshold for answering
        allow_conflicts: Whether to proceed if numeric conflicts detected
        
    Returns:
        AnswerResult with answer text, citations, and metadata
    """
    # 1) Grounding: confidence & conflicts
    conf = score_confidence(hits)
    conflict, conflict_meta = detect_numeric_conflicts(hits, query)
    weak = conf < min_confidence
    conflict_flag = (conflict and not allow_conflicts)

    if weak or conflict_flag:
        # Still pack a tiny sources table to show provenance
        ctx, sources = pack_context(hits[:2], token_budget=300, sentence_overlap=0, prefer_summary=True)
        msg = "Insufficient evidence to answer confidently."
        if conflict_flag:
            msg += " Sources appear to conflict."
        if not sources:
            return AnswerResult(text=msg, sources_table=[])
        
        # Produce a short explanation with a citation to [1]
        txt = f"{msg} See sources for details. [1]"
        return AnswerResult(
            text=txt, 
            citations=[], 
            sources_table=sources, 
            trace={
                "confidence": conf, 
                "conflict": conflict, 
                "conflict_meta": conflict_meta,
                "grounding_failed": True,
                "reason": "weak_confidence" if weak else "conflicts_detected"
            }
        )

    # 2) Pack context with budget & anti-dup
    context_str, sources_table = pack_context(
        hits, 
        token_budget=token_budget, 
        sentence_overlap=1, 
        prefer_summary=True
    )

    # 3) Delegate to answerer (LLM or rule-based)
    result = answerer.answer(query=query, packed_context=context_str, sources_table=sources_table)
    
    # 4) Attach trace information
    result.trace.update({
        "confidence": conf, 
        "conflict": conflict,
        "grounding_passed": True,
        "context_tokens": len(context_str.split()) * 1.3,  # Rough token estimate
        "sources_used": len(sources_table)
    })
    
    return result


def synthesize_answer_simple(
    query: str,
    hits: List[RetrievalHit],
    answerer: LlmAnswerer,
    **kwargs
) -> AnswerResult:
    """
    Simplified wrapper for synthesize_answer with common defaults.
    """
    return synthesize_answer(
        query=query,
        hits=hits,
        answerer=answerer,
        **kwargs
    )
