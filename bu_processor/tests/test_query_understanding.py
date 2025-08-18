# tests/test_query_understanding.py
import pytest
from bu_processor.query.models import ChatTurn
from bu_processor.query.heuristic_rewriter import HeuristicRewriter
from bu_processor.query.heuristic_expander import HeuristicExpander
from bu_processor.query.pipeline import QueryPipeline

from bu_processor.embeddings.testing_backend import FakeDeterministicEmbeddings
from bu_processor.index.faiss_index import FaissIndex
from bu_processor.storage.sqlite_store import SQLiteStore
from bu_processor.pipeline.upsert_pipeline import embed_and_index_chunks
from bu_processor.retrieval.dense import DenseKnnRetriever
from bu_processor.retrieval.bm25 import Bm25Index
from bu_processor.retrieval.hybrid import HybridRetriever

def _seed(store, embedder, index):
    chunks = [
        {"text": "Professional liability insurance covers financial losses from negligence.", "page": 1, "section": "Insurance"},
        {"text": "Pet insurance may include vet visits.", "page": 2, "section": "Insurance"},
        {"text": "Corporate finance optimizes capital structure and funding.", "page": 3, "section": "Finance"},
        {"text": "Cats are small domestic animals and love to sleep.", "page": 4, "section": "Pets"},
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
    store = SQLiteStore(url=f"sqlite:///{tmp_path/'q.db'}")
    embedder = FakeDeterministicEmbeddings(dim=64)
    index = FaissIndex()
    _seed(store, embedder, index)
    yield store, embedder, index
    store.close()

def test_heuristic_rewriter_basic():
    """Test heuristic rewriter extracts meaningful query from chat"""
    chat = [
        ChatTurn(role="user", content="Hi there, need help with insurance."),
        ChatTurn(role="assistant", content="What exactly?"),
        ChatTurn(role="user", content="Which insurance covers financial loss from professional errors?"),
    ]
    rw = HeuristicRewriter()
    result = rw.rewrite(chat)
    
    assert isinstance(result, str)
    assert len(result) > 0
    # Should extract key terms and remove stop words
    assert "insurance" in result.lower()
    assert "financial" in result.lower() or "loss" in result.lower()
    # Should not contain greetings/stop words
    assert "hi" not in result.lower()
    assert "there" not in result.lower()

def test_heuristic_rewriter_deterministic():
    """Test that rewriter produces consistent results"""
    chat = [
        ChatTurn(role="user", content="Hello! Please help me with corporate finance questions."),
        ChatTurn(role="assistant", content="Sure, what would you like to know?"),
        ChatTurn(role="user", content="How do companies optimize their capital structure?"),
    ]
    rw = HeuristicRewriter()
    
    result1 = rw.rewrite(chat)
    result2 = rw.rewrite(chat)
    
    assert result1 == result2
    assert "companies" in result1.lower() or "corporate" in result1.lower()
    assert "capital" in result1.lower()

def test_heuristic_rewriter_empty_chat():
    """Test rewriter handles empty or no-user chats gracefully"""
    rw = HeuristicRewriter()
    
    # Empty chat
    assert rw.rewrite([]) == ""
    
    # No user messages
    chat_no_user = [
        ChatTurn(role="assistant", content="Hello, how can I help?"),
        ChatTurn(role="system", content="Be helpful."),
    ]
    assert rw.rewrite(chat_no_user) == ""

def test_heuristic_expander_basic():
    """Test heuristic expander generates diverse paraphrases"""
    ex = HeuristicExpander()
    
    query = "insurance financial loss"
    expansions = ex.expand(query, num=3)
    
    assert len(expansions) >= 2
    assert all(isinstance(exp, str) for exp in expansions)
    # Should have some template-based expansions
    assert any("explain" in exp.lower() for exp in expansions)
    # Should have synonym-based expansions
    assert any("coverage" in exp.lower() or "policy" in exp.lower() for exp in expansions)

def test_heuristic_expander_deterministic():
    """Test that expander produces consistent results"""
    ex = HeuristicExpander()
    
    query = "corporate finance capital structure"
    
    result1 = ex.expand(query, num=2)
    result2 = ex.expand(query, num=2)
    
    assert result1 == result2
    assert len(result1) == 2

def test_heuristic_expander_deduplication():
    """Test that expander removes duplicates"""
    ex = HeuristicExpander()
    
    query = "simple query"
    expansions = ex.expand(query, num=10)  # Request many
    
    # Should be deduplicated
    assert len(expansions) == len(set(expansions))

def test_query_plan_all_queries_property():
    """Test QueryPlan.all_queries deduplication and ordering"""
    from bu_processor.query.models import QueryPlan
    
    plan = QueryPlan(
        focused_query="insurance coverage",
        expanded_queries=["coverage insurance", "insurance coverage", "policy coverage"],
    )
    
    all_q = plan.all_queries
    assert len(all_q) == 3  # Deduplicated
    assert all_q[0] == "insurance coverage"  # Focused first
    assert "coverage insurance" in all_q
    assert "policy coverage" in all_q

def test_rewriter_and_expander_deterministic(tmp_path):
    """Test complete pipeline with rewriter and expander is deterministic"""
    chat = [
        ChatTurn(role="user", content="Hi there, need help with insurance."),
        ChatTurn(role="assistant", content="What exactly?"),
        ChatTurn(role="user", content="Which insurance covers financial loss from professional errors?"),
    ]
    rw = HeuristicRewriter()
    ex = HeuristicExpander()
    qp = QueryPipeline(rw, ex, enable_rewrite=True, enable_expand=True, expansions_k=2)

    plan = qp.build_plan(chat)
    assert isinstance(plan.focused_query, str) and len(plan.focused_query) > 0
    assert len(plan.expanded_queries) >= 2
    # stable, non-empty and deduped
    assert len(plan.all_queries) >= 3
    assert len(set(plan.all_queries)) == len(plan.all_queries)

def test_query_pipeline_flags():
    """Test pipeline respects enable/disable flags"""
    chat = [
        ChatTurn(role="user", content="Test query about insurance"),
    ]
    rw = HeuristicRewriter()
    ex = HeuristicExpander()
    
    # Rewrite disabled
    qp1 = QueryPipeline(rw, ex, enable_rewrite=False, enable_expand=True, expansions_k=2)
    plan1 = qp1.build_plan(chat)
    assert plan1.focused_query == "Test query about insurance"  # Fallback to original
    assert len(plan1.expanded_queries) >= 1  # Expansion still works
    
    # Expansion disabled
    qp2 = QueryPipeline(rw, ex, enable_rewrite=True, enable_expand=False, expansions_k=2)
    plan2 = qp2.build_plan(chat)
    assert len(plan2.expanded_queries) == 0  # No expansions
    
    # Both disabled
    qp3 = QueryPipeline(rw, ex, enable_rewrite=False, enable_expand=False, expansions_k=2)
    plan3 = qp3.build_plan(chat)
    assert plan3.focused_query == "Test query about insurance"
    assert len(plan3.expanded_queries) == 0

def test_multi_query_retrieval_union_rrf(corpus):
    """Test multi-query retrieval with RRF fusion"""
    store, embedder, index = corpus
    
    dense = DenseKnnRetriever(embedder, index, store)
    bm25 = Bm25Index(store); bm25.build_from_store()
    hybrid = HybridRetriever(dense=dense, bm25=bm25, embedder=embedder,
                             fusion="rrf", use_mmr=False, reranker=None, summarizer=None)

    chat = [
        ChatTurn(role="user", content="Hello"),
        ChatTurn(role="assistant", content="Hi, how can I help?"),
        ChatTurn(role="user", content="Which insurance covers financial loss due to negligence?"),
    ]
    qp = QueryPipeline(HeuristicRewriter(), HeuristicExpander(), enable_rewrite=True, enable_expand=True, expansions_k=2)
    plan = qp.build_plan(chat)

    # union retrieval across focused + expansions, then RRF
    hits = qp.retrieve_union(plan, hybrid, top_k_per_query=4, final_top_k=3)
    assert len(hits) == 3
    # Top results should be insurance-related
    assert any("insurance" in h.text.lower() for h in hits)

def test_query_pipeline_with_none_components():
    """Test pipeline works when rewriter/expander are None"""
    chat = [
        ChatTurn(role="user", content="Test insurance query"),
    ]
    
    qp = QueryPipeline(rewriter=None, expander=None, enable_rewrite=True, enable_expand=True, expansions_k=2)
    plan = qp.build_plan(chat)
    
    # Should fall back to last user message
    assert plan.focused_query == "Test insurance query"
    assert len(plan.expanded_queries) == 0
    assert plan.trace["rewriter"] == "fallback_last_user"
    assert plan.trace["expander"] == "disabled_or_none"

def test_factories_query_rewriter_heuristic():
    """Test factory creates heuristic rewriter correctly"""
    from bu_processor.config import Settings
    from bu_processor.factories import make_query_rewriter
    
    settings = Settings(QUERY_REWRITER_BACKEND="heuristic")
    
    import bu_processor.factories
    original_settings = bu_processor.factories.settings
    bu_processor.factories.settings = settings
    
    try:
        rewriter = make_query_rewriter()
        assert rewriter is not None
        assert isinstance(rewriter, HeuristicRewriter)
    finally:
        bu_processor.factories.settings = original_settings

def test_factories_query_expander_heuristic():
    """Test factory creates heuristic expander correctly"""
    from bu_processor.config import Settings
    from bu_processor.factories import make_query_expander
    
    settings = Settings(QUERY_EXPANDER_BACKEND="heuristic")
    
    import bu_processor.factories
    original_settings = bu_processor.factories.settings
    bu_processor.factories.settings = settings
    
    try:
        expander = make_query_expander()
        assert expander is not None
        assert isinstance(expander, HeuristicExpander)
    finally:
        bu_processor.factories.settings = original_settings

def test_factories_query_pipeline():
    """Test factory creates complete query pipeline"""
    from bu_processor.config import Settings
    from bu_processor.factories import make_query_pipeline
    
    settings = Settings(
        QUERY_REWRITER_BACKEND="heuristic",
        QUERY_EXPANDER_BACKEND="heuristic",
        ENABLE_QUERY_REWRITE=True,
        ENABLE_QUERY_EXPANSION=True,
        QUERY_EXPANSIONS_K=3
    )
    
    import bu_processor.factories
    original_settings = bu_processor.factories.settings
    bu_processor.factories.settings = settings
    
    try:
        pipeline = make_query_pipeline()
        assert pipeline is not None
        assert isinstance(pipeline, QueryPipeline)
        assert pipeline.enable_rewrite is True
        assert pipeline.enable_expand is True
        assert pipeline.expansions_k == 3
    finally:
        bu_processor.factories.settings = original_settings

def test_factories_none_backends():
    """Test factory handles 'none' backends correctly"""
    from bu_processor.config import Settings
    from bu_processor.factories import make_query_rewriter, make_query_expander
    
    settings = Settings(
        QUERY_REWRITER_BACKEND="none",
        QUERY_EXPANDER_BACKEND="none"
    )
    
    import bu_processor.factories
    original_settings = bu_processor.factories.settings
    bu_processor.factories.settings = settings
    
    try:
        rewriter = make_query_rewriter()
        expander = make_query_expander()
        assert rewriter is None
        assert expander is None
    finally:
        bu_processor.factories.settings = original_settings
