from typing import List, Protocol, Optional, Dict, Any, TYPE_CHECKING
import numpy as np

# Import the RetrievalHit model for reranker interface
if TYPE_CHECKING:
    from bu_processor.retrieval.models import RetrievalHit
    from bu_processor.query.models import ChatTurn
    from bu_processor.answering.models import AnswerResult
else:
    RetrievalHit = Any
    ChatTurn = Any
    AnswerResult = Any

class EmbeddingsBackend(Protocol):
    """Protocol for embedding generation backends."""
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for processing
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        ...
    
    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings produced by this backend."""
        ...

class VectorIndex(Protocol):
    """Protocol for vector similarity search backends."""
    
    def create(self, dim: int, metric: str = "cosine", namespace: Optional[str] = None) -> None:
        """Create/initialize the vector index.
        
        Args:
            dim: Embedding dimension
            metric: Distance metric ("cosine", "euclidean", etc.)
            namespace: Optional namespace for multi-tenant indexes
        """
        ...
    
    def upsert(self, ids: List[str], vectors: np.ndarray, metadata: List[Dict[str, Any]],
               namespace: Optional[str] = None) -> None:
        """Insert or update vectors in the index.
        
        Args:
            ids: List of unique identifiers for vectors
            vectors: numpy array of shape (len(ids), embedding_dim)
            metadata: List of metadata dictionaries for each vector
            namespace: Optional namespace
        """
        ...
    
    def query(self, vector: np.ndarray, top_k: int = 10, namespace: Optional[str] = None,
              metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            vector: Query vector of shape (embedding_dim,)
            top_k: Number of results to return
            namespace: Optional namespace to search in
            metadata_filter: Optional metadata filtering
            
        Returns:
            List of results with keys: id, score, metadata
        """
        ...
    
    def delete(self, ids: List[str], namespace: Optional[str] = None) -> None:
        """Delete vectors by IDs.
        
        Args:
            ids: List of vector IDs to delete
            namespace: Optional namespace
        """
        ...


class Reranker(Protocol):
    """Protocol for reranking search results by relevance to query."""
    
    def rerank(self, query: str, hits: List["RetrievalHit"], top_k: Optional[int] = None) -> List["RetrievalHit"]:
        """
        Return the same hits re-ordered by true relevance to the query.
        May truncate to top_k if provided.
        
        Args:
            query: The search query
            hits: List of retrieval hits to rerank
            top_k: Optional limit on number of results to return
            
        Returns:
            Reranked list of hits, possibly truncated
        """
        ...


class QueryAwareSummarizer(Protocol):
    """Protocol for query-aware text summarization."""
    
    def summarize(self, query: str, text: str, target_tokens: int = 160) -> str:
        """
        Produce a compact, query-relevant snippet from text. Must be deterministic and fast.
        
        Args:
            query: The search query for context
            text: The text to summarize
            target_tokens: Target number of tokens in summary
            
        Returns:
            Query-aware summary text
        """
        ...


class QueryRewriter(Protocol):
    """Protocol for rewriting multi-turn chat into focused search queries."""
    
    def rewrite(self, chat: List["ChatTurn"]) -> str:
        """
        Condense multi-turn chat conversation into a focused search query.
        
        Args:
            chat: List of chat turns (user, assistant, system)
            
        Returns:
            Focused search query string
        """
        ...


class QueryExpander(Protocol):
    """Protocol for expanding queries into multiple paraphrases."""
    
    def expand(self, focused_query: str, num: int = 2) -> List[str]:
        """
        Generate diverse paraphrases of the focused query.
        
        Args:
            focused_query: The main query to expand
            num: Number of expansions to generate
            
        Returns:
            List of expanded/paraphrased queries
        """
        ...


class LlmAnswerer(Protocol):
    """Protocol for LLM-based answer generation with citations."""
    
    def answer(self, query: str, packed_context: str, sources_table: List[Dict[str, Any]]) -> "AnswerResult":
        """
        Generate an answer with citations from packed context.
        
        Args:
            query: The user's question
            packed_context: Pre-processed context with source markers [1], [2], etc.
            sources_table: Metadata for each source (1-based indexing)
            
        Returns:
            AnswerResult with text, citations, and source metadata
        """
        ...
