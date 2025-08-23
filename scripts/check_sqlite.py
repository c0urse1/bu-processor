#!/usr/bin/env python3
"""
💾 SQLITE DATABASE VALIDATION SCRIPT
===================================

Validates SQLite database content and structure for the BU-Processor system.
Shows statistics about stored documents and chunks.
"""

import sqlite3
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configuration
DB_PATH = Path("bu_store.db")
SAMPLE_LIMIT = 5

def connect_database(db_path: Path) -> sqlite3.Connection:
    """Connect to SQLite database"""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn

def check_database_structure(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    """Check database structure and tables"""
    print("🔍 Checking database structure...")
    
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"📊 Found {len(tables)} tables: {', '.join(tables)}")
    
    # Get schema for each table
    table_schemas = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [f"{row[1]} ({row[2]})" for row in cursor.fetchall()]
        table_schemas[table] = columns
        print(f"   📋 {table}: {len(columns)} columns")
        for col in columns:
            print(f"      - {col}")
    
    return table_schemas

def get_documents_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Get statistics about documents table"""
    print(f"\n📄 Document Statistics:")
    
    cursor = conn.cursor()
    
    # Basic counts
    cursor.execute("SELECT COUNT(*) FROM documents")
    total_docs = cursor.fetchone()[0]
    print(f"   Total documents: {total_docs:,}")
    
    if total_docs == 0:
        return {"total": 0}
    
    # Get date range
    cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM documents")
    min_date, max_date = cursor.fetchone()
    if min_date:
        print(f"   Date range: {min_date} to {max_date}")
    
    # Get sources
    cursor.execute("SELECT source, COUNT(*) as count FROM documents WHERE source IS NOT NULL GROUP BY source ORDER BY count DESC")
    sources = cursor.fetchall()
    if sources:
        print(f"   Sources:")
        for source, count in sources:
            print(f"     - {source}: {count:,} documents")
    
    # Check for metadata
    cursor.execute("SELECT COUNT(*) FROM documents WHERE meta IS NOT NULL AND meta != ''")
    docs_with_meta = cursor.fetchone()[0]
    print(f"   Documents with metadata: {docs_with_meta:,}")
    
    return {
        "total": total_docs,
        "with_metadata": docs_with_meta,
        "sources": dict(sources) if sources else {}
    }

def get_chunks_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Get statistics about chunks table"""
    print(f"\n📦 Chunk Statistics:")
    
    cursor = conn.cursor()
    
    # Basic counts
    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cursor.fetchone()[0]
    print(f"   Total chunks: {total_chunks:,}")
    
    if total_chunks == 0:
        return {"total": 0}
    
    # Average chunks per document
    cursor.execute("""
        SELECT AVG(chunk_count) as avg_chunks
        FROM (SELECT doc_id, COUNT(*) as chunk_count FROM chunks GROUP BY doc_id)
    """)
    avg_chunks = cursor.fetchone()[0]
    if avg_chunks:
        print(f"   Average chunks per document: {avg_chunks:.1f}")
    
    # Text length statistics
    cursor.execute("SELECT AVG(LENGTH(text)), MIN(LENGTH(text)), MAX(LENGTH(text)) FROM chunks")
    avg_len, min_len, max_len = cursor.fetchone()
    if avg_len:
        print(f"   Text length - Avg: {avg_len:.0f}, Min: {min_len}, Max: {max_len}")
    
    # Page distribution
    cursor.execute("SELECT page, COUNT(*) as count FROM chunks WHERE page IS NOT NULL GROUP BY page ORDER BY count DESC LIMIT 10")
    page_dist = cursor.fetchall()
    if page_dist:
        print(f"   Top pages by chunk count:")
        for page, count in page_dist[:5]:
            print(f"     - Page {page}: {count:,} chunks")
    
    return {
        "total": total_chunks,
        "avg_per_document": avg_chunks,
        "avg_text_length": avg_len
    }

def show_sample_documents(conn: sqlite3.Connection, limit: int = SAMPLE_LIMIT):
    """Show sample documents"""
    print(f"\n📋 Sample Documents (limit: {limit}):")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT doc_id, title, source, created_at, meta 
        FROM documents 
        ORDER BY created_at DESC 
        LIMIT ?
    """, (limit,))
    
    docs = cursor.fetchall()
    
    if not docs:
        print("   No documents found")
        return
    
    for i, doc in enumerate(docs, 1):
        print(f"\n   {i}. Document ID: {doc['doc_id']}")
        print(f"      Title: {doc['title'] or 'N/A'}")
        print(f"      Source: {doc['source'] or 'N/A'}")
        print(f"      Created: {doc['created_at']}")
        
        # Parse metadata if available
        if doc['meta']:
            try:
                metadata = json.loads(doc['meta']) if isinstance(doc['meta'], str) else doc['meta']
                if isinstance(metadata, dict):
                    print(f"      Metadata keys: {list(metadata.keys())}")
                    
                    # Show interesting metadata
                    for key in ['file_name', 'classification', 'file_hash', 'text_length']:
                        if key in metadata:
                            value = metadata[key]
                            if key == 'classification' and isinstance(value, dict):
                                print(f"        {key}: {value.get('predicted_label')} (conf: {value.get('confidence', 'N/A')})")
                            else:
                                print(f"        {key}: {value}")
            except Exception as e:
                print(f"      Metadata: (parse error: {e})")

def show_sample_chunks(conn: sqlite3.Connection, limit: int = SAMPLE_LIMIT):
    """Show sample chunks"""
    print(f"\n📝 Sample Chunks (limit: {limit}):")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.chunk_id, c.doc_id, c.page, c.section, 
               LEFT(c.text, 100) as text_preview,
               LENGTH(c.text) as text_length,
               d.title as doc_title
        FROM chunks c
        LEFT JOIN documents d ON c.doc_id = d.doc_id
        ORDER BY c.created_at DESC
        LIMIT ?
    """, (limit,))
    
    chunks = cursor.fetchall()
    
    if not chunks:
        print("   No chunks found")
        return
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n   {i}. Chunk ID: {chunk['chunk_id']}")
        print(f"      Document: {chunk['doc_title'] or 'N/A'} ({chunk['doc_id']})")
        if chunk['page']:
            print(f"      Page: {chunk['page']}")
        if chunk['section']:
            print(f"      Section: {chunk['section']}")
        print(f"      Length: {chunk['text_length']} chars")
        print(f"      Preview: {chunk['text_preview']}...")

def check_data_integrity(conn: sqlite3.Connection) -> Dict[str, bool]:
    """Check data integrity"""
    print(f"\n🔍 Data Integrity Checks:")
    
    cursor = conn.cursor()
    issues = {}
    
    # Check for orphaned chunks
    cursor.execute("""
        SELECT COUNT(*) FROM chunks c 
        LEFT JOIN documents d ON c.doc_id = d.doc_id 
        WHERE d.doc_id IS NULL
    """)
    orphaned_chunks = cursor.fetchone()[0]
    if orphaned_chunks > 0:
        print(f"   ⚠️  Found {orphaned_chunks} orphaned chunks (no matching document)")
        issues['orphaned_chunks'] = True
    else:
        print(f"   ✅ No orphaned chunks found")
        issues['orphaned_chunks'] = False
    
    # Check for documents without chunks
    cursor.execute("""
        SELECT COUNT(*) FROM documents d 
        LEFT JOIN chunks c ON d.doc_id = c.doc_id 
        WHERE c.doc_id IS NULL
    """)
    docs_without_chunks = cursor.fetchone()[0]
    if docs_without_chunks > 0:
        print(f"   ⚠️  Found {docs_without_chunks} documents without chunks")
        issues['docs_without_chunks'] = True
    else:
        print(f"   ✅ All documents have chunks")
        issues['docs_without_chunks'] = False
    
    # Check for empty text chunks
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE text IS NULL OR text = ''")
    empty_chunks = cursor.fetchone()[0]
    if empty_chunks > 0:
        print(f"   ⚠️  Found {empty_chunks} chunks with empty text")
        issues['empty_chunks'] = True
    else:
        print(f"   ✅ No empty text chunks found")
        issues['empty_chunks'] = False
    
    return issues

def main():
    """Main validation function"""
    print("💾 SQLite Database Validation Script")
    print("=" * 50)
    
    try:
        # Connect to database
        print(f"🔧 Connecting to database: {DB_PATH}")
        conn = connect_database(DB_PATH)
        print("✅ Database connection successful")
        
        # Check structure
        table_schemas = check_database_structure(conn)
        
        # Get statistics
        doc_stats = get_documents_stats(conn)
        chunk_stats = get_chunks_stats(conn)
        
        # Show samples if data exists
        if doc_stats.get('total', 0) > 0:
            show_sample_documents(conn)
            
        if chunk_stats.get('total', 0) > 0:
            show_sample_chunks(conn)
        
        # Check integrity
        integrity_issues = check_data_integrity(conn)
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"   Documents: {doc_stats.get('total', 0):,}")
        print(f"   Chunks: {chunk_stats.get('total', 0):,}")
        
        if any(integrity_issues.values()):
            print(f"   ⚠️  Data integrity issues found")
        else:
            print(f"   ✅ No data integrity issues")
        
        if doc_stats.get('total', 0) == 0:
            print(f"\n💡 Database is empty. To add data:")
            print(f"   python scripts/bulk_ingest.py")
        
        print(f"\n✅ SQLite validation completed!")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)
    
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
