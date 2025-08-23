from __future__ import annotations
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import uuid4
import json
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Text, JSON, DateTime
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.engine import Engine
from sqlalchemy.sql import select, insert, update

class SQLiteStore:
    def __init__(self, url: str = "sqlite:///bu_store.db"):
        self.engine: Engine = create_engine(url, future=True)
        self.md = MetaData()
        self.documents = Table(
            "documents", self.md,
            Column("doc_id", String, primary_key=True),
            Column("title", String, nullable=True),
            Column("source", String, nullable=True),
            Column("created_at", DateTime, default=datetime.utcnow),
            Column("meta", SQLITE_JSON, nullable=True),
        )
        self.chunks = Table(
            "chunks", self.md,
            Column("chunk_id", String, primary_key=True),
            Column("doc_id", String, nullable=False),
            Column("page", Integer, nullable=True),
            Column("section", String, nullable=True),
            Column("text", Text, nullable=False),
            Column("meta", SQLITE_JSON, nullable=True),
            Column("created_at", DateTime, default=datetime.utcnow),
        )
        self.md.create_all(self.engine)

    def add_document(self, content: str, metadata: Dict[str, Any], source: str) -> str:
        """Add a new document with content and metadata"""
        doc_id = str(uuid4())
        
        with self.engine.begin() as conn:
            conn.execute(insert(self.documents).values(
                doc_id=doc_id,
                title=metadata.get('filename', source),
                source=source,
                meta=metadata
            ))
            
            # Add content as a single chunk for now
            chunk_id = str(uuid4())
            conn.execute(insert(self.chunks).values(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=content,
                meta=metadata
            ))
        
        return doc_id

    def update_document_metadata(self, doc_id: str, additional_metadata: Dict[str, Any]) -> bool:
        """Update document metadata with additional information"""
        with self.engine.begin() as conn:
            # Get current metadata
            result = conn.execute(
                select(self.documents.c.meta).where(self.documents.c.doc_id == doc_id)
            ).first()
            
            if not result:
                return False
            
            # Merge metadata
            current_meta = result[0] or {}
            updated_meta = {**current_meta, **additional_metadata}
            
            # Update document
            conn.execute(
                update(self.documents)
                .where(self.documents.c.doc_id == doc_id)
                .values(meta=updated_meta)
            )
            
            # Also update chunk metadata
            conn.execute(
                update(self.chunks)
                .where(self.chunks.c.doc_id == doc_id)
                .values(meta=updated_meta)
            )
        
        return True

    def upsert_document(self, doc_id: Optional[str], title: Optional[str], source: Optional[str],
                        meta: Optional[Dict[str, Any]] = None) -> str:
        doc_id = doc_id or str(uuid4())
        with self.engine.begin() as conn:
            exists = conn.execute(select(self.documents.c.doc_id).where(self.documents.c.doc_id == doc_id)).first()
            if exists:
                conn.execute(update(self.documents).where(self.documents.c.doc_id == doc_id)
                             .values(title=title, source=source, meta=meta))
            else:
                conn.execute(insert(self.documents).values(doc_id=doc_id, title=title, source=source, meta=meta))
        return doc_id

    def upsert_chunks(self, doc_id: str, chunk_records: List[Dict[str, Any]]) -> List[str]:
        """
        chunk_records: [{chunk_id?, text, page?, section?, meta?}, ...]
        returns list of chunk_ids
        """
        ids = []
        with self.engine.begin() as conn:
            for rec in chunk_records:
                cid = rec.get("chunk_id") or str(uuid4())
                conn.execute(insert(self.chunks).values(
                    chunk_id=cid,
                    doc_id=doc_id,
                    page=rec.get("page"),
                    section=rec.get("section"),
                    text=rec["text"],
                    meta=rec.get("meta"),
                ))
                ids.append(cid)
        return ids

    def search_documents(self, query: str = "", limit: int = 10, 
                        predicted_label: str = None, min_confidence: float = None) -> List[Dict[str, Any]]:
        """
        Search documents with optional filtering by classification results.
        
        Args:
            query: Text search query (searches in title and source)
            limit: Maximum number of results
            predicted_label: Filter by predicted classification label
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of document records with metadata
        """
        with self.engine.begin() as conn:
            # Build query
            stmt = select(self.documents)
            
            # Text search
            if query.strip():
                search_term = f"%{query}%"
                stmt = stmt.where(
                    (self.documents.c.title.like(search_term)) |
                    (self.documents.c.source.like(search_term))
                )
            
            # Classification filters
            if predicted_label:
                stmt = stmt.where(
                    self.documents.c.meta['predicted_label'].astext == predicted_label
                )
            
            if min_confidence is not None:
                stmt = stmt.where(
                    self.documents.c.meta['predicted_confidence'].astext.cast(float) >= min_confidence
                )
            
            # Order by creation date (newest first) and limit
            stmt = stmt.order_by(self.documents.c.created_at.desc()).limit(limit)
            
            result = conn.execute(stmt)
            return [dict(row._mapping) for row in result]
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """
        Get classification statistics from stored documents.
        
        Returns:
            Dict with label counts, confidence stats, etc.
        """
        with self.engine.begin() as conn:
            # Get all documents with classification metadata
            stmt = select(self.documents.c.meta).where(
                self.documents.c.meta['predicted_label'].is_not(None)
            )
            
            result = conn.execute(stmt)
            documents = [row[0] for row in result]
            
            if not documents:
                return {"total_documents": 0, "labels": {}, "confidence": {}}
            
            # Analyze data
            label_counts = {}
            confidence_scores = []
            
            for metadata in documents:
                if not metadata:
                    continue
                    
                # Count labels
                label = metadata.get('predicted_label')
                if label:
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                # Collect confidence scores
                confidence = metadata.get('predicted_confidence')
                if confidence is not None:
                    confidence_scores.append(float(confidence))
            
            # Calculate confidence statistics
            confidence_stats = {}
            if confidence_scores:
                confidence_stats = {
                    "mean": sum(confidence_scores) / len(confidence_scores),
                    "min": min(confidence_scores),
                    "max": max(confidence_scores),
                    "count": len(confidence_scores)
                }
            
            return {
                "total_documents": len(documents),
                "labels": label_counts,
                "confidence": confidence_stats
            }
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID"""
        with self.engine.begin() as conn:
            result = conn.execute(
                select(self.documents).where(self.documents.c.doc_id == doc_id)
            ).first()
            
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.begin() as conn:
            row = conn.execute(select(self.chunks).where(self.chunks.c.chunk_id == chunk_id)).first()
            if not row: 
                return None
            return dict(row._mapping)
    
    def close(self):
        """Close the database connection."""
        self.engine.dispose()
