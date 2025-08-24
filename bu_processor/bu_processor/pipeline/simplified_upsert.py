# bu_processor/pipeline/simplified_upsert.py
"""
Simplified upsert pipeline using the new Embedder and PineconeManager.
This replaces the complex pipeline with MVP-focused functionality.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import os
from datetime import datetime, timezone

from ..embeddings.embedder import Embedder
from ..integrations.pinecone_facade import make_pinecone_manager
from ..storage.sqlite_store import SQLiteStore
from ..core.logging_setup import get_logger

if TYPE_CHECKING:
    from ..integrations.pinecone_facade import PineconeManager

logger = get_logger("simplified_upsert")

class SimplifiedUpsertPipeline:
    """
    Simplified upsert pipeline for MVP.
    Combines text chunking, embedding generation, and vector storage.
    """
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        pinecone_manager: Optional["PineconeManager"] = None,
        sqlite_store: Optional[SQLiteStore] = None,
        namespace: Optional[str] = None
    ):
        self.embedder = embedder or Embedder()
        self.namespace = namespace or os.getenv("PINECONE_NAMESPACE")
        
        # Initialize PineconeManager using standardized wiring
        self.pinecone_manager = pinecone_manager or make_pinecone_manager(
            index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV"),   # v2
            cloud=os.getenv("PINECONE_CLOUD"),       # v3
            region=os.getenv("PINECONE_REGION"),     # v3
            namespace=self.namespace
        )
        
        # Ensure index exists with correct dimension
        self.pinecone_manager.ensure_index(dimension=self.embedder.dimension)
        
        # SQLite store for metadata persistence
        self.sqlite_store = sqlite_store or SQLiteStore()
        
        logger.info(
            "SimplifiedUpsertPipeline initialized",
            embedding_model=self.embedder.model_name,
            embedding_dimension=self.embedder.dimension,
            index_name=self.pinecone_manager.index_name,
            namespace=self.namespace
        )
    
    def upsert_document(
        self,
        doc_id: Optional[str],
        doc_title: Optional[str],
        doc_source: Optional[str],
        chunks: List[Dict[str, Any]],
        doc_meta: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simplified document upsert with chunking and vectorization.
        
        Args:
            doc_id: Document ID (if None, will be generated)
            doc_title: Document title
            doc_source: Document source/path
            chunks: List of text chunks with metadata
            doc_meta: Additional document metadata
            namespace: Pinecone namespace override
            
        Returns:
            Dict with operation results
        """
        use_namespace = namespace or self.namespace
        ingested_at = datetime.now(timezone.utc).isoformat()
        
        logger.info(
            "Starting document upsert",
            doc_id=doc_id,
            doc_title=doc_title,
            chunks_count=len(chunks),
            namespace=use_namespace
        )
        
        # 1. Store document metadata in SQLite
        enhanced_doc_meta = dict(doc_meta or {})
        enhanced_doc_meta["ingested_at"] = ingested_at
        
        final_doc_id = self.sqlite_store.upsert_document(
            doc_id=doc_id,
            title=doc_title,
            source=doc_source,
            meta=enhanced_doc_meta
        )
        
        # 2. Store chunks in SQLite and get chunk IDs
        enhanced_chunks = []
        for chunk in chunks:
            enhanced_chunk = dict(chunk)
            if "meta" not in enhanced_chunk:
                enhanced_chunk["meta"] = {}
            enhanced_chunk["meta"]["ingested_at"] = ingested_at
            enhanced_chunks.append(enhanced_chunk)
        
        chunk_ids = self.sqlite_store.upsert_chunks(
            final_doc_id,
            [{
                "text": c["text"],
                "page": c.get("page"),
                "section": c.get("section"),
                "meta": c.get("meta", {})
            } for c in enhanced_chunks]
        )
        
        # 3. Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        logger.info("Generating embeddings", texts_count=len(texts))
        vectors = self.embedder.encode(texts)
        
        # 4. Prepare metadata for Pinecone
        pinecone_metadata = []
        for chunk_id, chunk in zip(chunk_ids, enhanced_chunks):
            chunk_meta = chunk.get("meta", {}) or {}
            
            metadata = {
                "doc_id": final_doc_id,
                "chunk_id": chunk_id,
                "title": doc_title or "",
                "source": doc_source or "",
                "ingested_at": ingested_at,
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                # Include all chunk metadata
                **chunk_meta
            }
            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}
            pinecone_metadata.append(metadata)
        
        # 5. Upsert to Pinecone
        logger.info("Upserting to Pinecone", vectors_count=len(vectors))
        upsert_result = self.pinecone_manager.upsert_vectors(
            ids=chunk_ids,
            vectors=vectors,
            metadatas=pinecone_metadata,
            namespace=use_namespace
        )
        
        result = {
            "doc_id": final_doc_id,
            "chunk_ids": chunk_ids,
            "chunks_count": len(chunks),
            "embedding_dimension": self.embedder.dimension,
            "ingested_at": ingested_at,
            "pinecone_result": upsert_result,
            "namespace": use_namespace
        }
        
        logger.info("Document upsert completed", **result)
        return result
    
    def query_similar_documents(
        self,
        query_text: str,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query for similar documents using text input.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            namespace: Pinecone namespace override
            filter_dict: Metadata filters
            
        Returns:
            List of search results
        """
        use_namespace = namespace or self.namespace
        
        logger.info(
            "Querying similar documents",
            query_text=query_text[:100] + "..." if len(query_text) > 100 else query_text,
            top_k=top_k,
            namespace=use_namespace
        )
        
        return self.pinecone_manager.query_by_text(
            text=query_text,
            embedder=self.embedder,
            top_k=top_k,
            namespace=use_namespace,
            filter=filter_dict
        )

# Factory function for compatibility
def make_simplified_upsert_pipeline(**kwargs) -> SimplifiedUpsertPipeline:
    """Factory function to create SimplifiedUpsertPipeline."""
    return SimplifiedUpsertPipeline(**kwargs)
