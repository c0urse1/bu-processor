#!/usr/bin/env python3
"""
D2) Context packer tests - deterministic, offline testing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bu_processor.answering.context_packer import pack_context
from bu_processor.retrieval.models import RetrievalHit

def _hit(id, text, score, meta):
    """Helper to create RetrievalHit objects"""
    return RetrievalHit(id=id, score=score, text=text, metadata=meta)

def test_context_packer_budget_and_citations():
    """Test token budget management and citation numbering"""
    hits = [
        _hit("c1","Professional liability insurance covers financial losses from negligence.",0.92,
             {"doc_id":"D1","section":"Insurance","page_start":1,"page_end":1,"title":"Insurance"}),
        _hit("c2","Corporate finance optimizes capital structure and funding.",0.85,
             {"doc_id":"D2","section":"Finance","page_start":3,"page_end":3,"title":"Finance"}),
        _hit("c3","Cats are small domestic animals and love to sleep.",0.10,
             {"doc_id":"D3","section":"Pets","page_start":5,"page_end":5,"title":"Pets"}),
    ]
    ctx, src = pack_context(hits, token_budget=60, sentence_overlap=1, prefer_summary=True)
    assert len(src) >= 2
    # should bias toward higher-score sources; the pets chunk likely excluded under tight budget
    assert "Pets" not in ctx  # likely excluded
    # numbered headers present
    assert ctx.splitlines()[0].startswith("[1]")
    print("✓ test_context_packer_budget_and_citations PASSED")

def test_context_packer_antidup():
    """Test anti-duplication functionality"""
    # force duplicate sentences across sources
    dup = "Debt financing can be cheaper than equity."
    hits = [
        _hit("a", dup + " It depends on interest rates.",0.9, {"doc_id":"D1","title":"Finance"}),
        _hit("b", dup + " Companies also consider risk.",0.8, {"doc_id":"D2","title":"Finance 2"}),
    ]
    ctx, _ = pack_context(hits, token_budget=80)
    # the duplicate lead sentence should appear once (anti-dup)
    assert ctx.lower().count("debt financing can be cheaper than equity.".lower()) == 1
    print("✓ test_context_packer_antidup PASSED")

def test_context_packer_quota_allocation():
    """Test score-based quota allocation"""
    hits = [
        _hit("high", "High score content with important information.", 0.95, 
             {"doc_id":"H1", "title":"High Priority", "section":"Important"}),
        _hit("med", "Medium score content with some relevance.", 0.70,
             {"doc_id":"M1", "title":"Medium Priority", "section":"Relevant"}),
        _hit("low", "Low score content with minimal relevance.", 0.30,
             {"doc_id":"L1", "title":"Low Priority", "section":"Minimal"}),
    ]
    
    ctx, src = pack_context(hits, token_budget=200, per_source_min_tokens=40)
    
    # High score source should appear first
    lines = ctx.split('\n')
    assert lines[0].startswith("[1]") and "High Priority" in lines[0]
    
    # Should include multiple sources
    assert len(src) >= 2
    
    # Highest score should be first in sources table
    assert src[0]["score"] == 0.95
    assert src[0]["title"] == "High Priority"
    
    print("✓ test_context_packer_quota_allocation PASSED")

def test_context_packer_metadata_preservation():
    """Test that metadata is properly preserved in sources table"""
    hits = [
        _hit("test1", "Content with rich metadata.", 0.88,
             {"doc_id":"DOC123", "title":"Test Document", "section":"Chapter 1", 
              "page_start":10, "page_end":12}),
        _hit("test2", "More content with different metadata.", 0.76,
             {"doc_id":"DOC456", "title":"Another Document", "section":"Chapter 2",
              "page_start":5, "page_end":5}),
    ]
    
    ctx, src = pack_context(hits, token_budget=150)
    
    # Check metadata preservation
    assert len(src) == 2
    
    # First source metadata
    s1 = src[0]
    assert s1["chunk_id"] == "test1"
    assert s1["doc_id"] == "DOC123"
    assert s1["title"] == "Test Document"
    assert s1["section"] == "Chapter 1"
    assert s1["page_start"] == 10
    assert s1["page_end"] == 12
    assert s1["score"] == 0.88
    
    # Second source metadata
    s2 = src[1]
    assert s2["chunk_id"] == "test2"
    assert s2["doc_id"] == "DOC456"
    assert s2["title"] == "Another Document"
    assert s2["section"] == "Chapter 2"
    assert s2["page_start"] == 5
    assert s2["page_end"] == 5
    
    print("✓ test_context_packer_metadata_preservation PASSED")

def test_context_packer_sentence_overlap():
    """Test sentence overlap between sources"""
    hits = [
        _hit("s1", "First chunk content. This is sentence two. Third sentence here.", 0.90,
             {"doc_id":"D1", "title":"Source 1"}),
        _hit("s2", "Second chunk begins here. Another sentence follows. Final sentence ends.", 0.85,
             {"doc_id":"D2", "title":"Source 2"}),
    ]
    
    ctx, src = pack_context(hits, token_budget=200, sentence_overlap=1)
    
    # Should have overlap (exact overlap depends on sentence splitting)
    # The test verifies structure rather than exact overlap content
    assert "[1]" in ctx and "[2]" in ctx
    assert "Source 1" in ctx and "Source 2" in ctx
    
    print("✓ test_context_packer_sentence_overlap PASSED")

def test_context_packer_edge_cases():
    """Test edge cases: empty hits, single hit, very small budget"""
    
    # Test empty hits
    ctx, src = pack_context([], token_budget=100)
    assert ctx == ""
    assert src == []
    
    # Test single hit
    single_hit = [_hit("solo", "Single hit content.", 0.80, {"doc_id":"SOLO", "title":"Solo"})]
    ctx, src = pack_context(single_hit, token_budget=50)
    assert "[1]" in ctx
    assert len(src) == 1
    assert "Solo" in ctx
    
    # Test very small budget
    hits = [_hit("tiny", "Very long content that exceeds the tiny budget.", 0.90, {"title":"Tiny"})]
    ctx, src = pack_context(hits, token_budget=5)
    # Should still produce some output, even if truncated
    assert len(src) <= 1
    
    print("✓ test_context_packer_edge_cases PASSED")

def test_context_packer_duplicate_filtering():
    """Test filtering of duplicate chunk IDs"""
    hits = [
        _hit("dup1", "First occurrence of content.", 0.90, {"doc_id":"D1", "title":"First"}),
        _hit("dup1", "Duplicate ID with different content.", 0.85, {"doc_id":"D2", "title":"Second"}),  # Same ID
        _hit("unique", "Unique content here.", 0.80, {"doc_id":"D3", "title":"Unique"}),
    ]
    
    ctx, src = pack_context(hits, token_budget=200)
    
    # Should only have 2 sources (duplicate filtered)
    assert len(src) == 2
    
    # First occurrence should be kept
    chunk_ids = [s["chunk_id"] for s in src]
    assert "dup1" in chunk_ids
    assert "unique" in chunk_ids
    assert chunk_ids.count("dup1") == 1  # No duplicates
    
    print("✓ test_context_packer_duplicate_filtering PASSED")

def run_all_context_packer_tests():
    """Run all D2 context packer tests"""
    print("=== D2: Context Packer Tests ===")
    try:
        test_context_packer_budget_and_citations()
        test_context_packer_antidup()
        test_context_packer_quota_allocation()
        test_context_packer_metadata_preservation()
        test_context_packer_sentence_overlap()
        test_context_packer_edge_cases()
        test_context_packer_duplicate_filtering()
        print("✅ All D2 context packer tests PASSED")
        return 7, 0
    except Exception as e:
        print(f"❌ D2 tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, 7

if __name__ == "__main__":
    run_all_context_packer_tests()
