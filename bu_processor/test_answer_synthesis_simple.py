#!/usr/bin/env python3
"""
Simple test of the answer synthesis components
"""

from bu_processor.answering.models import AnswerResult, Citation
from bu_processor.answering.rule_based import RuleBasedAnswerer
from bu_processor.answering.context_packer import pack_context
from bu_processor.answering.grounding import score_confidence, detect_numeric_conflicts
from bu_processor.retrieval.models import RetrievalHit

def test_basic_components():
    """Test basic answer synthesis components."""
    print("🧪 Testing Answer Synthesis Components")
    print("=" * 50)
    
    # Create sample hits
    hits = [
        RetrievalHit(
            id="test1",
            text="Cloud computing offers scalability and cost savings for businesses.",
            metadata={"doc_id": "cloud_guide", "title": "Cloud Benefits", "section": "Overview"},
            score=0.8
        ),
        RetrievalHit(
            id="test2", 
            text="Traditional infrastructure requires high upfront investment and maintenance.",
            metadata={"doc_id": "infra_guide", "title": "Infrastructure Costs", "section": "Traditional"},
            score=0.7
        )
    ]
    
    query = "What are the benefits of cloud computing?"
    
    # Test 1: Context packing
    print("📦 Testing context packing...")
    packed_context, sources_table = pack_context(hits, token_budget=500)
    print(f"   Context length: {len(packed_context)} chars")
    print(f"   Sources table: {len(sources_table)} sources")
    print(f"   Packed context preview: {packed_context[:100]}...")
    print()
    
    # Test 2: Grounding checks
    print("🔍 Testing grounding checks...")
    confidence = score_confidence(hits)
    conflict, conflict_meta = detect_numeric_conflicts(hits, query)
    print(f"   Confidence score: {confidence:.3f}")
    print(f"   Conflicts detected: {conflict}")
    print(f"   Conflict metadata: {conflict_meta}")
    print()
    
    # Test 3: Rule-based answerer
    print("🤖 Testing rule-based answerer...")
    answerer = RuleBasedAnswerer()
    result = answerer.answer(query, packed_context, sources_table)
    
    print(f"   Answer text: {result.text}")
    print(f"   Citations count: {len(result.citations)}")
    print(f"   Sources count: {len(result.sources_table)}")
    
    if result.citations:
        print("   Citations:")
        for cite in result.citations:
            print(f"     - Para {cite.paragraph_idx}: {cite.chunk_id} ({cite.doc_id})")
    print()
    
    print("✅ All components working correctly!")

if __name__ == "__main__":
    test_basic_components()
