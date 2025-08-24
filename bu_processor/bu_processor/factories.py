from bu_processor.config import settings
from bu_processor.storage.sqlite_store import SQLiteStore

# =============================================================================
# NEW SIMPLIFIED FACTORIES FOR MVP
# =============================================================================

def make_simplified_embedder():
    """Create the new simplified embedder."""
    from bu_processor.embeddings.embedder import Embedder
    return Embedder()

def make_simplified_pinecone_manager():
    """Create the new simplified Pinecone manager using standardized wiring."""
    from bu_processor.integrations.pinecone_facade import make_pinecone_manager
    import os
    
    return make_pinecone_manager(
        index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),       # v2
        cloud=os.getenv("PINECONE_CLOUD"),           # v3
        region=os.getenv("PINECONE_REGION"),         # v3
        namespace=os.getenv("PINECONE_NAMESPACE")
    )

def make_simplified_upsert_pipeline():
    """Create the new simplified upsert pipeline."""
    from bu_processor.pipeline.simplified_upsert import SimplifiedUpsertPipeline
    return SimplifiedUpsertPipeline()

# =============================================================================
# LEGACY FACTORIES (for backward compatibility)
# =============================================================================

# embeddings
def make_embedder():
    if settings.EMBEDDINGS_BACKEND == "sbert":
        from bu_processor.embeddings.sbert_backend import SbertEmbeddings
        return SbertEmbeddings(model_name=settings.SBERT_MODEL)
    elif settings.EMBEDDINGS_BACKEND == "openai":
        from bu_processor.embeddings.openai_backend import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY,
                                model=settings.OPENAI_EMBED_MODEL)
    elif settings.EMBEDDINGS_BACKEND == "fake":
        from bu_processor.embeddings.testing_backend import FakeDeterministicEmbeddings
        return FakeDeterministicEmbeddings(dim=128)
    elif settings.EMBEDDINGS_BACKEND == "simplified":
        # New simplified backend
        return make_simplified_embedder()
    else:
        raise ValueError("Unknown EMBEDDINGS_BACKEND")

# index
def make_index():
    if settings.VECTOR_INDEX == "faiss":
        from bu_processor.index.faiss_index import FaissIndex
        return FaissIndex()
    elif settings.VECTOR_INDEX == "pinecone":
        from bu_processor.index.pinecone_index import PineconeIndex
        return PineconeIndex(api_key=settings.PINECONE_API_KEY,
                             index_name=settings.PINECONE_INDEX)
    else:
        raise ValueError("Unknown VECTOR_INDEX")

def make_store():
    return SQLiteStore(url=settings.SQLITE_URL)

# add retriever factories
from bu_processor.config import settings

def make_dense_retriever():
    from bu_processor.factories import make_embedder, make_index, make_store
    from bu_processor.retrieval.dense import DenseKnnRetriever
    embedder = make_embedder()
    index = make_index()
    store = make_store()
    return DenseKnnRetriever(embedder=embedder, index=index, store=store, namespace=settings.NAMESPACE)

def make_bm25_index():
    from bu_processor.factories import make_store
    from bu_processor.retrieval.bm25 import Bm25Index
    store = make_store()
    bm25 = Bm25Index(store)
    # Note: build_from_store() should be called after documents are indexed
    return bm25

def make_reranker():
    name = settings.RERANKER_BACKEND
    if name == "cross_encoder":
        from bu_processor.rerank.cross_encoder_reranker import CrossEncoderReranker
        device = settings.CROSS_ENCODER_DEVICE or None
        return CrossEncoderReranker(model_name=settings.CROSS_ENCODER_MODEL, device=device)
    elif name == "heuristic":
        from bu_processor.rerank.testing_reranker import HeuristicOverlapReranker
        return HeuristicOverlapReranker()
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown RERANKER_BACKEND={name}")

def make_summarizer():
    name = settings.SUMMARIZER_BACKEND
    if name == "extractive":
        from bu_processor.summarize.query_aware_extractive import QueryAwareExtractiveSummarizer
        return QueryAwareExtractiveSummarizer(max_sentences=3)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown SUMMARIZER_BACKEND={name}")

def make_hybrid_retriever():
    from bu_processor.factories import make_embedder, make_reranker, make_summarizer
    from bu_processor.retrieval.hybrid import HybridRetriever
    dense = make_dense_retriever()
    bm25 = make_bm25_index()
    # Build BM25 index from current store contents
    bm25.build_from_store()
    embedder = make_embedder()
    reranker = make_reranker()
    summarizer = make_summarizer()
    return HybridRetriever(
        dense=dense, bm25=bm25, embedder=embedder,
        fusion="rrf", alpha_dense=0.6, alpha_bm25=0.4,
        use_mmr=True, mmr_lambda=0.65,
        reranker=reranker,
        summarizer=summarizer,
        summary_tokens=settings.SUMMARY_TOKENS,
    )

def make_query_rewriter():
    name = settings.QUERY_REWRITER_BACKEND
    if name == "heuristic":
        from bu_processor.query.heuristic_rewriter import HeuristicRewriter
        return HeuristicRewriter()
    if name == "openai":
        from bu_processor.query.llm_rewriter import OpenAIRewriter
        return OpenAIRewriter(api_key=settings.OPENAI_API_KEY)
    if name == "none":
        return None
    raise ValueError(f"Unknown QUERY_REWRITER_BACKEND={name}")

def make_query_expander():
    name = settings.QUERY_EXPANDER_BACKEND
    if name == "heuristic":
        from bu_processor.query.heuristic_expander import HeuristicExpander
        return HeuristicExpander()
    if name == "openai":
        from bu_processor.query.llm_expander import OpenAIExpander
        return OpenAIExpander(api_key=settings.OPENAI_API_KEY)
    if name == "none":
        return None
    raise ValueError(f"Unknown QUERY_EXPANDER_BACKEND={name}")

def make_query_pipeline():
    from bu_processor.query.pipeline import QueryPipeline
    return QueryPipeline(
        rewriter=make_query_rewriter(),
        expander=make_query_expander(),
        enable_rewrite=settings.ENABLE_QUERY_REWRITE,
        enable_expand=settings.ENABLE_QUERY_EXPANSION,
        expansions_k=settings.QUERY_EXPANSIONS_K,
    )


def make_answerer():
    """Create LLM answerer based on configuration."""
    name = settings.ANSWERER_BACKEND
    
    if name == "rule_based":
        from bu_processor.answering.rule_based_answerer import RuleBasedAnswerer
        return RuleBasedAnswerer()
    
    if name == "rule_based_simple":
        from bu_processor.answering.rule_based import RuleBasedAnswerer
        return RuleBasedAnswerer()
    
    if name == "openai":
        from bu_processor.answering.openai_answerer import OpenAIAnswerer
        return OpenAIAnswerer(
            model=settings.OPENAI_ANSWERER_MODEL,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=settings.OPENAI_ANSWERER_MAX_TOKENS
        )
    
    if name == "openai_simple":
        from bu_processor.answering.llm_openai import OpenAiAnswerer
        return OpenAiAnswerer(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_ANSWERER_MODEL
        )
    
    if name == "none":
        return None
    
    raise ValueError(f"Unknown ANSWERER_BACKEND={name}")


def make_answer_pipeline():
    """Create complete answer synthesis pipeline."""
    from bu_processor.answering.pipeline import AnswerPipeline
    
    answerer = make_answerer()
    if not answerer:
        return None
    
    return AnswerPipeline(
        answerer=answerer,
        token_budget=settings.ANSWER_TOKEN_BUDGET,
        min_confidence=settings.ANSWER_MIN_CONFIDENCE,
        min_sources=settings.ANSWER_MIN_SOURCES,
        prefer_summary=settings.ANSWER_PREFER_SUMMARY
    )


def make_synthesize_orchestrator():
    """Create the synthesize orchestrator function configured with settings."""
    from bu_processor.answering.synthesize import synthesize_answer
    
    answerer = make_answerer()
    if not answerer:
        return None
    
    def configured_synthesize(query: str, hits, **kwargs):
        """Pre-configured synthesize function."""
        defaults = {
            "token_budget": settings.ANSWER_TOKEN_BUDGET,
            "min_confidence": settings.ANSWER_MIN_CONFIDENCE,
            "allow_conflicts": settings.ANSWER_ALLOW_CONFLICTS
        }
        defaults.update(kwargs)
        
        return synthesize_answer(
            query=query,
            hits=hits,
            answerer=answerer,
            **defaults
        )
    
    return configured_synthesize


def synthesize(query: str, hits):
    """Simple synthesize function that uses current settings."""
    from bu_processor.answering.synthesize import synthesize_answer
    
    answerer = make_answerer()
    if not answerer:
        raise ValueError("No answerer configured")
    
    return synthesize_answer(
        query=query,
        hits=hits,
        answerer=answerer,
        token_budget=settings.ANSWER_TOKEN_BUDGET,
        min_confidence=settings.ANSWER_MIN_CONFIDENCE,
        allow_conflicts=settings.ANSWER_ALLOW_CONFLICTS,
    )
