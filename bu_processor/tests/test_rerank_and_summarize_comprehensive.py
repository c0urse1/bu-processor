# tests/test_rerank_and_summarize_comprehensive.py
import os
import pytest
from bu_processor.embeddings.testing_backend import FakeDeterministicEmbeddings
from bu_processor.index.faiss_index import FaissIndex
from bu_processor.storage.sqlite_store import SQLiteStore
from bu_processor.pipeline.upsert_pipeline import embed_and_index_chunks
from bu_processor.retrieval.dense import DenseKnnRetriever
from bu_processor.retrieval.bm25 import Bm25Index
from bu_processor.retrieval.hybrid import HybridRetriever
from bu_processor.rerank.testing_reranker import HeuristicOverlapReranker
from bu_processor.summarize.query_aware_extractive import QueryAwareExtractiveSummarizer
from bu_processor.retrieval.models import RetrievalHit
from bu_processor.pipeline.post_retrieve import rerank_and_summarize

def seed(store, embedder, index):
    chunks = [
        {"text": "Professional liability insurance covers financial losses from mistakes.", "page": 1, "section": "Insurance"},
        {"text": "Corporate finance optimizes capital structure and project funding.", "page": 2, "section": "Finance"},
        {"text": "Cats are small domestic animals and love to sleep.", "page": 3, "section": "Pets"},
        {"text": "Pet insurance may cover vet visits and vaccinations.", "page": 4, "section": "Insurance"},
    ]
    embed_and_index_chunks(
        doc_title="Mini",
        doc_source="unit",
        doc_meta=None,
        chunks=chunks,
        embedder=embedder,
        index=index,
        store=store,
        namespace=None,
    )

@pytest.fixture
def corpus(tmp_path):
    store = SQLiteStore(url=f"sqlite:///{tmp_path/'r.db'}")
    embedder = FakeDeterministicEmbeddings(dim=64)
    index = FaissIndex()
    seed(store, embedder, index)
    yield store, embedder, index
    # Cleanup
    store.close()

@pytest.fixture
def sample_hits():
    """Sample retrieval hits for testing"""
    return [
        RetrievalHit(
            id="1",
            score=0.9,
            text="Professional liability insurance covers financial losses from lawsuits. This type of insurance protects businesses against claims.",
            metadata={"section": "Insurance", "doc_type": "policy"}
        ),
        RetrievalHit(
            id="2", 
            score=0.7,
            text="Pet insurance helps cover veterinary costs for cats and dogs. Many pet owners find this useful for unexpected medical bills.",
            metadata={"section": "Pets", "doc_type": "faq"}
        ),
        RetrievalHit(
            id="3",
            score=0.8,
            text="Corporate finance involves managing company capital structure and funding decisions. CFOs use various financial instruments.",
            metadata={"section": "Finance", "doc_type": "guide"}
        ),
        RetrievalHit(
            id="4",
            score=0.6,
            text="Cats are domestic animals that sleep frequently. They require regular veterinary care and proper nutrition for health.",
            metadata={"section": "Pets", "doc_type": "blog"}
        ),
    ]

def test_heuristic_reranker_changes_order(corpus):
    store, embedder, index = corpus
    dense = DenseKnnRetriever(embedder, index, store)
    bm25 = Bm25Index(store); bm25.build_from_store()

    # Intentionally pick a query that lexical/BM25 might not put first
    query = "Which insurance covers financial loss due to errors?"
    # Hybrid without reranker first
    plain_hybrid = HybridRetriever(dense=dense, bm25=bm25, embedder=embedder,
                                   fusion="rrf", use_mmr=False,
                                   reranker=None, summarizer=None)
    base_hits = plain_hybrid.retrieve(query, final_top_k=3)
    base_top_texts = [h.text for h in base_hits]

    # Now with heuristic reranker
    reranked_hybrid = HybridRetriever(dense=dense, bm25=bm25, embedder=embedder,
                                      fusion="rrf", use_mmr=False,
                                      reranker=HeuristicOverlapReranker(),
                                      summarizer=None)
    reranked_hits = reranked_hybrid.retrieve(query, final_top_k=3)
    reranked_top_texts = [h.text for h in reranked_hits]

    # Expect change in top order or at least top-1 match moving up
    assert base_top_texts != reranked_top_texts or "insurance" in reranked_top_texts[0].lower()

def test_query_aware_summarizer_adds_summary(corpus):
    store, embedder, index = corpus
    dense = DenseKnnRetriever(embedder, index, store)
    bm25 = Bm25Index(store); bm25.build_from_store()

    query = "financial losses insurance"
    hybrid = HybridRetriever(dense=dense, bm25=bm25, embedder=embedder,
                             fusion="rrf", use_mmr=False,
                             reranker=HeuristicOverlapReranker(),
                             summarizer=QueryAwareExtractiveSummarizer(),
                             )
    hits = hybrid.retrieve(query, final_top_k=2)
    assert len(hits) == 2
    assert "summary" in hits[0].metadata
    # summary should be shorter than full text and contain query terms where possible
    assert len(hits[0].metadata["summary"]) <= len(hits[0].text)

def test_heuristic_reranker_basic(sample_hits):
    """Test heuristic overlap reranker works and is deterministic"""
    reranker = HeuristicOverlapReranker()
    
    # Test insurance-related query
    query = "What insurance covers professional liability?"
    reranked = reranker.rerank(query, sample_hits.copy(), top_k=3)
    
    assert len(reranked) == 3
    # Should prioritize insurance-related content
    assert "insurance" in reranked[0].text.lower()
    assert reranked[0].id == "1"  # Professional liability should rank first
    
    # Check scores are added
    assert "ce_score" in reranked[0].metadata
    assert reranked[0].metadata["ce_score"] > 0

def test_heuristic_reranker_deterministic(sample_hits):
    """Test that reranker produces consistent results"""
    reranker = HeuristicOverlapReranker()
    query = "corporate finance capital structure"
    
    result1 = reranker.rerank(query, sample_hits.copy())
    result2 = reranker.rerank(query, sample_hits.copy())
    
    # Should be identical
    assert len(result1) == len(result2)
    for h1, h2 in zip(result1, result2):
        assert h1.id == h2.id
        assert h1.metadata["ce_score"] == h2.metadata["ce_score"]

def test_extractive_summarizer_basic():
    """Test query-aware extractive summarizer"""
    summarizer = QueryAwareExtractiveSummarizer(max_sentences=2)
    
    query = "What is professional liability insurance?"
    text = "Professional liability insurance covers financial losses from lawsuits. This type of insurance protects businesses against claims. Some companies also offer general liability coverage. The premiums vary by industry and risk factors."
    
    summary = summarizer.summarize(query, text, target_tokens=50)
    
    assert len(summary) > 0
    assert "professional" in summary.lower() or "liability" in summary.lower() or "insurance" in summary.lower()
    # Should be shorter than original
    assert len(summary) < len(text)

def test_extractive_summarizer_deterministic():
    """Test that summarizer is deterministic"""
    summarizer = QueryAwareExtractiveSummarizer(max_sentences=2)
    
    query = "pet insurance costs"
    text = "Pet insurance helps cover veterinary costs for cats and dogs. Many pet owners find this useful for unexpected medical bills. The monthly premiums range from $30 to $100. Coverage varies by provider and plan type."
    
    summary1 = summarizer.summarize(query, text, target_tokens=40)
    summary2 = summarizer.summarize(query, text, target_tokens=40)
    
    assert summary1 == summary2

def test_extractive_summarizer_empty_text():
    """Test summarizer handles empty/invalid input gracefully"""
    summarizer = QueryAwareExtractiveSummarizer()
    
    assert summarizer.summarize("query", "") == ""
    assert summarizer.summarize("", "text") != ""  # Should still return something

def test_post_retrieve_pipeline_rerank_only(sample_hits):
    """Test post-retrieve pipeline with reranking only"""
    reranker = HeuristicOverlapReranker()
    
    query = "financial liability coverage"
    result = rerank_and_summarize(
        query=query,
        hits=sample_hits.copy(),
        reranker=reranker,
        summarizer=None,
        top_k=2
    )
    
    assert len(result) == 2
    assert "ce_score" in result[0].metadata
    assert "summary" not in result[0].metadata  # No summarizer used
    # Should prioritize insurance content
    assert "insurance" in result[0].text.lower()

def test_post_retrieve_pipeline_summarize_only(sample_hits):
    """Test post-retrieve pipeline with summarization only"""
    summarizer = QueryAwareExtractiveSummarizer(max_sentences=1)
    
    query = "pet care and veterinary costs"
    result = rerank_and_summarize(
        query=query,
        hits=sample_hits.copy()[:2],  # Just first 2
        reranker=None,
        summarizer=summarizer,
        top_k=2
    )
    
    assert len(result) == 2
    assert "summary" in result[0].metadata
    assert len(result[0].metadata["summary"]) > 0
    assert "ce_score" not in result[0].metadata  # No reranker used

def test_post_retrieve_pipeline_full(sample_hits):
    """Test complete post-retrieve pipeline with both reranking and summarization"""
    reranker = HeuristicOverlapReranker()
    summarizer = QueryAwareExtractiveSummarizer(max_sentences=1)
    
    query = "insurance coverage for business liability"
    result = rerank_and_summarize(
        query=query,
        hits=sample_hits.copy(),
        reranker=reranker,
        summarizer=summarizer,
        summary_tokens=30,
        top_k=2
    )
    
    assert len(result) == 2
    
    # Should have both reranking scores and summaries
    for hit in result:
        assert "ce_score" in hit.metadata
        assert "summary" in hit.metadata
        assert len(hit.metadata["summary"]) > 0
    
    # Top result should be insurance-related due to reranking
    assert "insurance" in result[0].text.lower()

def test_factories_reranker_heuristic():
    """Test factory creates heuristic reranker correctly"""
    from bu_processor.config import Settings
    from bu_processor.factories import make_reranker
    
    # Mock settings for heuristic reranker
    settings = Settings(RERANKER_BACKEND="heuristic")
    
    # Temporarily override global settings
    import bu_processor.factories
    original_settings = bu_processor.factories.settings
    bu_processor.factories.settings = settings
    
    try:
        reranker = make_reranker()
        assert reranker is not None
        assert isinstance(reranker, HeuristicOverlapReranker)
    finally:
        bu_processor.factories.settings = original_settings

def test_factories_reranker_none():
    """Test factory returns None when reranker disabled"""
    from bu_processor.config import Settings
    from bu_processor.factories import make_reranker
    
    settings = Settings(RERANKER_BACKEND="none")
    
    import bu_processor.factories
    original_settings = bu_processor.factories.settings
    bu_processor.factories.settings = settings
    
    try:
        reranker = make_reranker()
        assert reranker is None
    finally:
        bu_processor.factories.settings = original_settings

def test_factories_summarizer_extractive():
    """Test factory creates extractive summarizer correctly"""
    from bu_processor.config import Settings
    from bu_processor.factories import make_summarizer
    
    settings = Settings(SUMMARIZER_BACKEND="extractive")
    
    import bu_processor.factories
    original_settings = bu_processor.factories.settings
    bu_processor.factories.settings = settings
    
    try:
        summarizer = make_summarizer()
        assert summarizer is not None
        assert isinstance(summarizer, QueryAwareExtractiveSummarizer)
    finally:
        bu_processor.factories.settings = original_settings

def test_factories_summarizer_none():
    """Test factory returns None when summarizer disabled"""
    from bu_processor.config import Settings
    from bu_processor.factories import make_summarizer
    
    settings = Settings(SUMMARIZER_BACKEND="none")
    
    import bu_processor.factories
    original_settings = bu_processor.factories.settings
    bu_processor.factories.settings = settings
    
    try:
        summarizer = make_summarizer()
        assert summarizer is None
    finally:
        bu_processor.factories.settings = original_settings

# Optional slow test with real CrossEncoder
@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("RUN_REAL_RERANKER"), reason="set RUN_REAL_RERANKER=1 to run")
def test_cross_encoder_reranker_smoke(corpus):
    """Test real CrossEncoder reranker (requires network/download)"""
    from bu_processor.rerank.cross_encoder_reranker import CrossEncoderReranker
    
    store, embedder, index = corpus
    dense = DenseKnnRetriever(embedder, index, store)
    bm25 = Bm25Index(store); bm25.build_from_store()
    
    hybrid = HybridRetriever(dense=dense, bm25=bm25, embedder=embedder,
                             fusion="rrf", use_mmr=False,
                             reranker=CrossEncoderReranker(), summarizer=None)
    hits = hybrid.retrieve("financial loss insurance", final_top_k=3)
    assert len(hits) >= 1
    assert "ce_score" in hits[0].metadata
    # CrossEncoder should return float scores
    assert isinstance(hits[0].metadata["ce_score"], float)
