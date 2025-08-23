#!/usr/bin/env python3
"""
💾 SIMPLE SQLITE DATABASE CHECK
==============================

Quick and simple SQLite database content check for BU-Processor.
Based on the original check pattern but with enhanced output.
"""

import sqlite3
from pathlib import Path

DB = Path("bu_store.db")

def main():
    """Simple SQLite database content check"""
    if not DB.exists():
        print("❌ bu_store.db nicht gefunden")
        print(f"   Expected location: {DB.absolute()}")
        return
    
    print("💾 SQLite Database Quick Check")
    print("=" * 40)
    
    try:
        con = sqlite3.connect(DB)
        cur = con.cursor()

        print("\n📊 [Documents]")
        cur.execute("SELECT COUNT(*) FROM documents")
        doc_count = cur.fetchone()[0]
        print(f"documents: {doc_count:,}")

        print("\n📦 [Chunks]")
        cur.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cur.fetchone()[0]
        print(f"chunks: {chunk_count:,}")
        
        if doc_count > 0:
            print(f"\n📄 [Beispiel-Dokument + 3 Chunks]")
            cur.execute("SELECT doc_id, title, substr(meta,1,120) FROM documents ORDER BY created_at DESC LIMIT 1")
            doc = cur.fetchone()
            
            if doc:
                doc_id = doc[0]
                title = doc[1] or "N/A"
                meta = doc[2] or ""
                
                print(f"doc_id: {doc_id}")
                print(f"title: {title}")
                print(f"meta: {meta}…")
                
                cur.execute("SELECT chunk_id, substr(text,1,120) FROM chunks WHERE doc_id=? LIMIT 3", (doc_id,))
                chunks = cur.fetchall()
                
                if chunks:
                    print(f"\nChunks ({len(chunks)}):")
                    for i, row in enumerate(chunks, 1):
                        chunk_id = row[0]
                        text_preview = row[1].replace("\n", " ").strip()
                        print(f" {i}. {chunk_id}: {text_preview}…")
                else:
                    print("   (Keine Chunks für dieses Dokument gefunden)")
        else:
            print("\n💡 Database ist leer")
            print("   Verwende: python scripts/bulk_ingest.py")
        
        # Quick stats summary
        if doc_count > 0 and chunk_count > 0:
            avg_chunks = chunk_count / doc_count
            print(f"\n📈 Quick Stats:")
            print(f"   Avg chunks per document: {avg_chunks:.1f}")
            
            # File size
            db_size = DB.stat().st_size / (1024 * 1024)  # MB
            print(f"   Database size: {db_size:.1f} MB")
        
        con.close()
        
    except sqlite3.Error as e:
        print(f"❌ SQLite error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
