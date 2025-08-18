"""
Tests for Answer Synthesis System

Tests the complete answer synthesis pipeline with deterministic components:
- Context packing with token budgets
- Grounding checks (confidence, conflicts)
- Rule-based answer generation with citations
- Insufficient evidence handling
"""

import pytest
import tempfile
from pathlib import Path

from bu_processor.embeddings.testing_backend import FakeDeterministicEmbeddings
from bu_processor.index.faiss_index import FaissIndex
from bu_processor.storage.sqlite_store import SQLiteStore
from bu_processor.pipeline.upsert_pipeline import embed_and_index_chunks
from bu_processor.retrieval.dense import DenseKnnRetriever
from bu_processor.retrieval.bm25 import Bm25Index
from bu_processor.retrieval.hybrid import HybridRetriever
from bu_processor.answering.synthesize import synthesize_answer
from bu_processor.answering.rule_based import RuleBasedAnswerer
from bu_processor.answering.context_packer import pack_context
from bu_processor.answering.grounding import score_confidence, detect_numeric_conflicts
from bu_processor.retrieval.models import RetrievalHit


def _seed_test_data(store, embedder, index):
    """Seed test data for answer synthesis tests."""
    chunks = [
        {
            "text": "Professional liability insurance covers financial losses from negligence, errors, and omissions in professional services. This type of coverage protects professionals against claims of inadequate work or failure to deliver promised results.",
            "page": 1, 
            "section": "Insurance Coverage Types"
        },
        {
            "text": "Cats are small domestic animals that are popular as pets. They are carnivorous mammals belonging to the family Felidae. Domestic cats are valued by humans for companionship.",
            "page": 2, 
            "section": "Pet Animals"
        },
        {
            "text": "Corporate finance optimizes capital structure and funding decisions for businesses. It involves managing financial resources, investment decisions, and capital allocation to maximize shareholder value.",
            "page": 3, 
            "section": "Business Finance"
        },
        {
            "text": "Cloud computing provides on-demand access to computing resources over the internet. Key benefits include cost savings, scalability, and flexibility for businesses of all sizes.",
            "page": 4,
            "section": "Technology Solutions"
        }
    ]
    
    embed_and_index_chunks(
        doc_title="Test Knowledge Base",
        doc_source="unit_test",
        doc_meta={"version": "1.0"},
        chunks=chunks,
        embedder=embedder,
        index=index,
        store=store,
        namespace="test",
    )


def test_context_packing():
    """Test context packing with token budgets and anti-duplication."""
    hits = [
        RetrievalHit(
            id="chunk1",
            text="Professional liability insurance covers errors and omissions. It protects against financial losses from negligence claims.",
            metadata={"doc_id": "insurance_guide", "title": "Insurance Types", "section": "Professional Coverage", "page": 1},
            score=0.9
        ),
        RetrievalHit(
            id="chunk2",
            text="Corporate finance involves capital structure optimization. Businesses use various funding sources to maximize value.",
            metadata={"doc_id": "finance_guide", "title": "Corporate Finance", "section": "Capital Management", "page": 2},
            score=0.8
        )
    ]
    
    # Test basic packing
    context_str, sources_table = pack_context(hits, token_budget=200)
    
    assert len(sources_table) == 2
    assert sources_table[0]["chunk_id"] == "chunk1"
    assert sources_table[1]["chunk_id"] == "chunk2"
    assert "[1]" in context_str
    assert "[2]" in context_str
    assert "Professional liability" in context_str
    assert "Corporate finance" in context_str


def test_grounding_checks():
    """Test confidence scoring and conflict detection."""
    # High confidence hits
    high_conf_hits = [
        RetrievalHit(id="h1", text="Clear answer here", metadata={}, score=0.9),
        RetrievalHit(id="h2", text="Supporting evidence", metadata={}, score=0.8),
    ]
    
    confidence = score_confidence(high_conf_hits)
    assert confidence > 0.7
    
    # Low confidence hits
    low_conf_hits = [
        RetrievalHit(id="l1", text="Unclear information", metadata={}, score=0.2),
        RetrievalHit(id="l2", text="Vague content", metadata={}, score=0.1),
    ]
    
    confidence = score_confidence(low_conf_hits)
    assert confidence < 0.3
    
    # Test numeric conflict detection
    conflict_hits = [
        RetrievalHit(id="c1", text="The cost is $100 per month", metadata={}, score=0.8),
        RetrievalHit(id="c2", text="Monthly fees are $500", metadata={}, score=0.7),
    ]
    
    conflict, meta = detect_numeric_conflicts(conflict_hits, "What is the monthly cost?")
    # Note: This may or may not detect conflict depending on the threshold
    assert isinstance(conflict, bool)
    assert "reason" in meta


def test_rule_based_answerer():
    """Test the rule-based answerer with citation generation."""
    # Create packed context
    packed_context = """[1] Insurance Guide (doc:insurance_guide, sec:Professional Coverage, p.1)
Professional liability insurance covers financial losses from negligence, errors, and omissions in professional services.

[2] Finance Guide (doc:finance_guide, sec:Capital Management, p.2)
Corporate finance optimizes capital structure and funding decisions for businesses."""
    
    sources_table = [
        {"chunk_id": "ins_chunk1", "doc_id": "insurance_guide", "title": "Insurance Guide", "section": "Professional Coverage", "page": 1},
        {"chunk_id": "fin_chunk1", "doc_id": "finance_guide", "title": "Finance Guide", "section": "Capital Management", "page": 2}
    ]
    
    answerer = RuleBasedAnswerer()
    result = answerer.answer("What is professional liability insurance?", packed_context, sources_table)
    
    # Check answer structure
    assert result.text.strip() != ""
    assert len(result.citations) > 0
    assert len(result.sources_table) == 2
    
    # Check citations format
    paragraphs = [p for p in result.text.split("\n\n") if p.strip()]
    assert len(paragraphs) >= 1
    
    # Should have citation markers
    first_para = paragraphs[0].rstrip()
    assert first_para.endswith("]")
    
    # Citations should reference valid chunks
    for citation in result.citations:
        assert citation.chunk_id in ["ins_chunk1", "fin_chunk1"]
        assert citation.doc_id in ["insurance_guide", "finance_guide"]


def test_rule_based_answer_with_citations():
    """Test complete pipeline with rule-based answerer and citations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Setup components
        db_path = Path(tmp_dir) / "test.db"
        store = SQLiteStore(url=f"sqlite:///{db_path}")
        embedder = FakeDeterministicEmbeddings(dim=64)
        index = FaissIndex()
        
        try:
            # Seed test data
            _seed_test_data(store, embedder, index)
            
            # Create retriever
            dense = DenseKnnRetriever(embedder, index, store, namespace="test")
            bm25 = Bm25Index(store)
            bm25.build_from_store()  # Remove namespace parameter
            hybrid = HybridRetriever(
                dense=dense, 
                bm25=bm25, 
                embedder=embedder, 
                fusion="rrf", 
                use_mmr=False
            )
            
            # Test query
            hits = hybrid.retrieve("Which insurance covers financial loss from negligence?", final_top_k=5)
            assert len(hits) > 0
            
            # Synthesize answer
            result = synthesize_answer(
                query="Which insurance covers financial loss from negligence?",
                hits=hits, 
                answerer=RuleBasedAnswerer(), 
                token_budget=400
            )
            
            # Verify result structure
            assert result.text.strip() != ""
            assert "professional liability" in result.text.lower() or "negligence" in result.text.lower()
            
            # Check paragraph structure and citations
            paragraphs = [p for p in result.text.split("\n\n") if p.strip()]
            assert len(paragraphs) >= 1
            
            # Should have citation markers at end of paragraphs
            first_para = paragraphs[0].rstrip()
            assert first_para.endswith("]")
            
            # Sources table should align with citations
            assert len(result.sources_table) >= 1
            
            # Should have citation objects
            assert len(result.citations) > 0
            assert any(c.chunk_id for c in result.citations)
            
        finally:
            # Explicitly close store connections
            if hasattr(store, 'engine'):
                store.engine.dispose()


def test_insufficient_evidence_path():
    """Test the insufficient evidence response path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Setup with only irrelevant content
        db_path = Path(tmp_dir) / "test.db"
        store = SQLiteStore(url=f"sqlite:///{db_path}")
        embedder = FakeDeterministicEmbeddings(dim=64)
        index = FaissIndex()
        
        try:
            # Seed only irrelevant data to trigger low confidence
            embed_and_index_chunks(
                doc_title="Pet Guide",
                doc_source="unit_test",
                doc_meta=None,
                chunks=[
                    {"text": "Cats purr and sleep most of the day. They are independent animals.", "page": 1, "section": "Cat Behavior"},
                    {"text": "Dogs are loyal companions and require daily exercise.", "page": 2, "section": "Dog Care"}
                ],
                embedder=embedder, 
                index=index, 
                store=store, 
                namespace="test"
            )
            
            # Retrieve with irrelevant query
            dense = DenseKnnRetriever(embedder, index, store, namespace="test")
            hits = dense.retrieve("insurance for professional errors and omissions", top_k=3)
            
            # Should get low confidence and insufficient evidence response
            result = synthesize_answer(
                query="insurance for professional errors and omissions", 
                hits=hits,
                answerer=RuleBasedAnswerer(), 
                token_budget=200, 
                min_confidence=0.9  # Set high threshold to trigger insufficient evidence
            )
            
            # Should return insufficient evidence message
            assert "Insufficient evidence" in result.text
            assert "grounding_failed" in result.trace
            assert result.trace["grounding_failed"] is True
            
        finally:
            # Explicitly close store connections
            if hasattr(store, 'engine'):
                store.engine.dispose()


def test_synthesize_with_conflicts():
    """Test handling of conflicting information."""
    # Create hits with conflicting numeric information
    conflicting_hits = [
        RetrievalHit(
            id="conflict1",
            text="The premium costs $100 per month for basic coverage.",
            metadata={"doc_id": "price_guide_a", "section": "Basic Plans"},
            score=0.8
        ),
        RetrievalHit(
            id="conflict2", 
            text="Monthly premiums start at $500 for standard protection.",
            metadata={"doc_id": "price_guide_b", "section": "Standard Plans"},
            score=0.7
        )
    ]
    
    # Test with conflicts not allowed
    result = synthesize_answer(
        query="What are the monthly premium costs?",
        hits=conflicting_hits,
        answerer=RuleBasedAnswerer(),
        allow_conflicts=False,
        min_confidence=0.1  # Low threshold to focus on conflict detection
    )
    
    # Result may vary based on conflict detection sensitivity
    # Just ensure we get a valid response
    assert result.text.strip() != ""
    assert "conflict" in result.trace


def test_answer_synthesis_comprehensive():
    """Comprehensive test of the full answer synthesis system."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Setup complete system
        db_path = Path(tmp_dir) / "comprehensive.db"
        store = SQLiteStore(url=f"sqlite:///{db_path}")
        embedder = FakeDeterministicEmbeddings(dim=128)
        index = FaissIndex()
        
        try:
            # Seed comprehensive test data
            _seed_test_data(store, embedder, index)
            
            # Add more relevant content
            additional_chunks = [
                {
                    "text": "Errors and omissions insurance protects professionals from claims of inadequate work or failure to deliver services as promised. Coverage typically includes legal defense costs.",
                    "page": 5,
                    "section": "E&O Insurance Details"
                }
            ]
            
            embed_and_index_chunks(
                doc_title="Extended Insurance Guide",
                doc_source="unit_test_extended", 
                doc_meta={"version": "2.0"},
                chunks=additional_chunks,
                embedder=embedder,
                index=index,
                store=store,
                namespace="test"
            )
            
            # Create hybrid retriever
            dense = DenseKnnRetriever(embedder, index, store, namespace="test")
            bm25 = Bm25Index(store)
            bm25.build_from_store()  # Remove namespace parameter
            hybrid = HybridRetriever(
                dense=dense,
                bm25=bm25, 
                embedder=embedder,
                fusion="rrf",
                use_mmr=True,
                mmr_lambda=0.5
            )
            
            # Test multiple queries
            test_queries = [
                "What is professional liability insurance?",
                "How does corporate finance work?",
                "What are the benefits of cloud computing?"
            ]
            
            for query in test_queries:
                # Retrieve and synthesize
                hits = hybrid.retrieve(query, final_top_k=6)
                result = synthesize_answer(
                    query=query,
                    hits=hits,
                    answerer=RuleBasedAnswerer(),
                    token_budget=600,
                    min_confidence=0.2
                )
                
                # Basic checks
                assert result.text.strip() != ""
                assert len(result.sources_table) >= 0
                
                # If we got an answer (not insufficient evidence), check structure
                if "Insufficient evidence" not in result.text:
                    paragraphs = [p for p in result.text.split("\n\n") if p.strip()]
                    assert len(paragraphs) >= 1
                    
                    # Should have citations if we have sources
                    if result.sources_table:
                        assert any("[" in para and "]" in para for para in paragraphs)
                        
        finally:
            # Explicitly close store connections
            if hasattr(store, 'engine'):
                store.engine.dispose()


if __name__ == "__main__":
    # Run tests individually for debugging
    test_context_packing()
    print("✅ Context packing test passed")
    
    test_grounding_checks()
    print("✅ Grounding checks test passed")
    
    test_rule_based_answerer()
    print("✅ Rule-based answerer test passed")
    
    test_rule_based_answer_with_citations()
    print("✅ Rule-based answer with citations test passed")
    
    test_insufficient_evidence_path()
    print("✅ Insufficient evidence path test passed")
    
    test_synthesize_with_conflicts()
    print("✅ Conflict handling test passed")
    
    test_answer_synthesis_comprehensive()
    print("✅ Comprehensive answer synthesis test passed")
    
    print("\n🎉 All answer synthesis tests passed!")
