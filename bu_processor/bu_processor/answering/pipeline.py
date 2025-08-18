from __future__ import annotations
from typing import List, Dict, Any
from bu_processor.retrieval.models import RetrievalHit
from bu_processor.answering.models import AnswerResult
from bu_processor.answering.context_packer import pack_context
from bu_processor.answering.grounding import check_grounding_quality, format_insufficient_evidence_response
from bu_processor.ports import LlmAnswerer

class AnswerPipeline:
    """
    Complete answer synthesis pipeline:
    retrieval hits → context packing → grounding checks → answer generation
    """
    
    def __init__(
        self,
        answerer: LlmAnswerer,
        token_budget: int = 1200,
        min_confidence: float = 0.3,
        min_sources: int = 2,
        prefer_summary: bool = True
    ):
        self.answerer = answerer
        self.token_budget = token_budget
        self.min_confidence = min_confidence
        self.min_sources = min_sources
        self.prefer_summary = prefer_summary
    
    def synthesize_answer(
        self, 
        query: str, 
        hits: List[RetrievalHit],
        force_answer: bool = False
    ) -> AnswerResult:
        """
        Generate a complete answer from retrieval hits.
        
        Args:
            query: User's question
            hits: Retrieved and reranked content chunks
            force_answer: If True, generate answer even if grounding is insufficient
            
        Returns:
            AnswerResult with answer text, citations, and source metadata
        """
        if not hits:
            return AnswerResult(
                text="No relevant sources found for your question.",
                citations=[],
                sources_table=[],
                trace={"method": "no_sources", "query": query}
            )
        
        # Step 1: Pack context with budget management
        packed_context, sources_table = pack_context(
            hits=hits,
            token_budget=self.token_budget,
            prefer_summary=self.prefer_summary
        )
        
        # Step 2: Check grounding quality
        is_sufficient, reason, warnings = check_grounding_quality(
            hits=hits,
            query=query,
            min_confidence=self.min_confidence,
            min_sources=self.min_sources
        )
        
        # Step 3: Generate answer or insufficient evidence response
        if is_sufficient or force_answer:
            # Generate answer with citations
            result = self.answerer.answer(query, packed_context, sources_table)
            
            # Add grounding warnings to trace
            if warnings:
                result.trace["grounding_warnings"] = warnings
                
            return result
        else:
            # Return insufficient evidence response
            insufficient_text = format_insufficient_evidence_response(
                query=query,
                reason=reason,
                warnings=warnings,
                sources_table=sources_table
            )
            
            return AnswerResult(
                text=insufficient_text,
                citations=[],
                sources_table=sources_table,
                trace={
                    "method": "insufficient_evidence",
                    "query": query,
                    "reason": reason,
                    "warnings": warnings,
                    "source_count": len(sources_table)
                }
            )
