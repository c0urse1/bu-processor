from __future__ import annotations
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import uuid4
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

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.begin() as conn:
            row = conn.execute(select(self.chunks).where(self.chunks.c.chunk_id == chunk_id)).first()
            if not row: 
                return None
            return dict(row._mapping)
    
    def close(self):
        """Close the database connection."""
        self.engine.dispose()
