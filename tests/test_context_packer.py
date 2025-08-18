from bu_processor.answering.context_packer import pack_context
from bu_processor.retrieval.models import RetrievalHit

def _hit(id, text, score, meta):
    return RetrievalHit(id=id, score=score, text=text, metadata=meta)

def test_context_packer_budget_and_citations():
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

def test_context_packer_antidup():
    # force duplicate sentences across sources
    dup = "Debt financing can be cheaper than equity."
    hits = [
        _hit("a", dup + " It depends on interest rates.",0.9, {"doc_id":"D1","title":"Finance"}),
        _hit("b", dup + " Companies also consider risk.",0.8, {"doc_id":"D2","title":"Finance 2"}),
    ]
    ctx, _ = pack_context(hits, token_budget=80)
    # the duplicate lead sentence should appear once (anti-dup)
    assert ctx.lower().count("debt financing can be cheaper than equity.".lower()) == 1

def test_context_packer_quota_allocation():
    """Test that higher-scored sources get more token allocation"""
    hits = [
        _hit("high_score", "High relevance content with important information. " * 10, 0.95,
             {"doc_id":"D1", "title":"High Score", "section":"Important"}),
        _hit("low_score", "Low relevance content with minimal value. " * 10, 0.3,
             {"doc_id":"D2", "title":"Low Score", "section":"Minor"}),
    ]
    
    ctx, sources = pack_context(hits, token_budget=200, per_source_min_tokens=30)
    
    # High score source should appear first and get more content
    assert sources[0]["score"] == 0.95  # Highest score first
    assert "High Score" in ctx
    
    # Context should contain more content from high-scoring source
    high_score_content = ctx.count("High relevance")
    low_score_content = ctx.count("Low relevance")
    assert high_score_content >= low_score_content

def test_context_packer_sentence_overlap():
    """Test sentence overlap for continuity"""
    hits = [
        _hit("chunk1", "First content piece. This has good information. More details follow.",
             0.9, {"doc_id":"D1", "title":"First", "section":"A"}),
        _hit("chunk2", "Second content piece. Different topic entirely. New information here.",
             0.8, {"doc_id":"D2", "title":"Second", "section":"B"}),
    ]
    
    ctx, _ = pack_context(hits, token_budget=200, sentence_overlap=1)
    
    # Should contain content from both chunks
    assert "First content" in ctx
    assert "Second content" in ctx
    
    # With overlap, some sentences might appear near boundaries
    lines = ctx.split('\n')
    assert len([line for line in lines if line.startswith('[')]) == 2  # Two sources

def test_context_packer_prefer_summary():
    """Test preference for summary over full text"""
    hits = [
        _hit("with_summary", "Full text is very long and detailed with lots of information that might be excessive.",
             0.9, {"doc_id":"D1", "title":"Doc", "summary": "Short summary of key points."}),
        _hit("no_summary", "Another piece of content without summary available.",
             0.8, {"doc_id":"D2", "title":"Doc2"}),
    ]
    
    ctx, sources = pack_context(hits, token_budget=100, prefer_summary=True)
    
    # Should use summary when available
    assert "Short summary of key points" in ctx
    # Should not use full text when summary is available
    assert "very long and detailed" not in ctx

def test_context_packer_unique_chunks():
    """Test that duplicate chunk IDs are filtered"""
    hits = [
        _hit("duplicate_id", "First version of content.", 0.9, {"doc_id":"D1", "title":"First"}),
        _hit("duplicate_id", "Second version of same chunk.", 0.8, {"doc_id":"D1", "title":"Second"}),
        _hit("unique_id", "Unique content here.", 0.85, {"doc_id":"D2", "title":"Unique"}),
    ]
    
    ctx, sources = pack_context(hits, token_budget=200)
    
    # Should only include one instance of duplicate ID (first one by order)
    assert len(sources) == 2
    assert sources[0]["chunk_id"] == "duplicate_id"
    assert sources[1]["chunk_id"] == "unique_id"
    
    # Should contain first version content
    assert "First version" in ctx
    assert "Second version" not in ctx

def test_context_packer_empty_hits():
    """Test handling of empty hit list"""
    ctx, sources = pack_context([], token_budget=100)
    
    assert ctx == ""
    assert sources == []

def test_context_packer_metadata_preservation():
    """Test that metadata is properly preserved in sources table"""
    hits = [
        _hit("meta_test", "Content with rich metadata.", 0.9, {
             "doc_id": "TEST_DOC_123",
             "title": "Test Document Title",
             "section": "Chapter 5: Analysis",
             "page_start": 15,
             "page_end": 17
         }),
    ]
    
    ctx, sources = pack_context(hits, token_budget=100)
    
    assert len(sources) == 1
    source = sources[0]
    
    # Verify all metadata is preserved
    assert source["chunk_id"] == "meta_test"
    assert source["doc_id"] == "TEST_DOC_123"
    assert source["title"] == "Test Document Title"
    assert source["section"] == "Chapter 5: Analysis"
    assert source["page_start"] == 15
    assert source["page_end"] == 17
    assert source["score"] == 0.9
    
    # Verify formatted context includes metadata
    assert "[1] Test Document Title" in ctx
    assert "doc:TEST_DOC_123" in ctx
    assert "sec:Chapter 5: Analysis" in ctx
    assert "p.15-17" in ctx
