from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    EMBEDDINGS_BACKEND: str = "sbert"     # sbert | openai | fake
    SBERT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"
    OPENAI_API_KEY: str = ""
    VECTOR_INDEX: str = "faiss"           # faiss | pinecone
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = "bu-index"
    SQLITE_URL: str = "sqlite:///bu_store.db"
    NAMESPACE: str = "default"
    
    # Reranker settings
    RERANKER_BACKEND: str = "cross_encoder"   # "none" | "cross_encoder" | "heuristic"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CROSS_ENCODER_DEVICE: str = ""            # "", "cpu", "cuda", etc.
    RERANK_TOP_K: int = 10
    
    # Summarizer settings
    SUMMARIZER_BACKEND: str = "extractive"    # "none" | "extractive"
    SUMMARY_TOKENS: int = 160
    
    # Query understanding settings
    QUERY_REWRITER_BACKEND: str = "heuristic"  # "none" | "heuristic" | "openai"
    QUERY_EXPANDER_BACKEND: str = "heuristic"  # "none" | "heuristic" | "openai"
    ENABLE_QUERY_REWRITE: bool = True
    ENABLE_QUERY_EXPANSION: bool = True
    QUERY_EXPANSIONS_K: int = 2

    # Answer Synthesis Settings
    ANSWERER_BACKEND: str = "rule_based"  # "rule_based" | "openai"
    ENABLE_ANSWER_SYNTHESIS: bool = True
    ANSWER_TOKEN_BUDGET: int = 1200
    ANSWER_MIN_CONFIDENCE: float = 0.25  # Updated to match user's spec
    ANSWER_MIN_SOURCES: int = 2
    ANSWER_PREFER_SUMMARY: bool = True
    ANSWER_ALLOW_CONFLICTS: bool = False  # New setting
    
    # OpenAI Answerer Settings (if using openai backend)
    OPENAI_ANSWERER_MODEL: str = "gpt-4o-mini"  # Updated to user's preferred model
    OPENAI_ANSWERER_MAX_TOKENS: int = 500
    OPENAI_ANSWER_MODEL: str = "gpt-4o-mini"  # Alternative name for compatibility

settings = Settings()
