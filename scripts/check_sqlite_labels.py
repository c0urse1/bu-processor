#!/usr/bin/env python3
"""
🗄️ SQLITE LABELS QUALITY CHECK
==============================

Quick check to verify that classification labels are properly stored
in SQLite metadata after bulk ingestion.
"""

import sqlite3
import json
import os
from pathlib import Path

def check_sqlite_labels():
    """Check SQLite database for classification metadata"""
    
    # Find database file
    db_paths = [
        Path("bu_store.db"),
        Path("bu_processor/bu_store.db"),
        Path("data/bu_store.db")
    ]
    
    db_path = None
    for path in db_paths:
        if path.exists():
            db_path = path
            break
    
    if not db_path:
        print("❌ SQLite database not found")
        print(f"   Searched in: {[str(p) for p in db_paths]}")
        return False
    
    print(f"🗄️ Checking SQLite database: {db_path}")
    
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Get table info
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        print(f"📋 Available tables: {', '.join(tables)}")
        
        if 'documents' not in tables:
            print("❌ No 'documents' table found")
            return False
        
        # Get document count
        cur.execute("SELECT COUNT(*) FROM documents")
        doc_count = cur.fetchone()[0]
        print(f"📄 Total documents: {doc_count:,}")
        
        if doc_count == 0:
            print("⚠️  No documents found in database")
            return True
        
        # Check recent documents with metadata
        print(f"\n🔍 Recent Documents with Classification Metadata:")
        print("-" * 60)
        
        cur.execute("""
            SELECT title, meta, created_at 
            FROM documents 
            ORDER BY rowid DESC 
            LIMIT 5
        """)
        
        for i, (title, meta_json, created_at) in enumerate(cur.fetchall(), 1):
            print(f"\n📄 Document {i}:")
            print(f"   Title: {title or 'N/A'}")
            print(f"   Created: {created_at}")
            
            if meta_json:
                try:
                    meta = json.loads(meta_json)
                    
                    # Check for classification metadata
                    classification_fields = [
                        'predicted_label', 'predicted_category', 
                        'predicted_confidence', 'classification_timestamp'
                    ]
                    
                    classification_meta = {k: v for k, v in meta.items() if k in classification_fields}
                    
                    if classification_meta:
                        print(f"   ✅ Classification metadata:")
                        for key, value in classification_meta.items():
                            if key == 'predicted_confidence' and isinstance(value, (int, float)):
                                print(f"      {key}: {value:.3f}")
                            else:
                                print(f"      {key}: {value}")
                    else:
                        print(f"   ❌ No classification metadata found")
                    
                    # Show other interesting metadata
                    other_fields = ['file_name', 'source', 'page_count', 'ingestion_method']
                    other_meta = {k: v for k, v in meta.items() if k in other_fields}
                    if other_meta:
                        print(f"   📋 Other metadata: {other_meta}")
                    
                    # Show first 100 chars of metadata for debugging
                    meta_preview = str(meta)[:200]
                    if len(str(meta)) > 200:
                        meta_preview += "..."
                    print(f"   🔍 Metadata preview: {meta_preview}")
                    
                except json.JSONDecodeError:
                    print(f"   ❌ Invalid JSON in metadata")
                    print(f"   Raw metadata: {meta_json[:200]}...")
            else:
                print(f"   ❌ No metadata found")
        
        # Summary statistics
        print(f"\n📊 Classification Statistics:")
        
        # Count documents with classification metadata
        cur.execute("""
            SELECT COUNT(*) FROM documents 
            WHERE meta LIKE '%predicted_label%'
        """)
        classified_count = cur.fetchone()[0]
        
        cur.execute("""
            SELECT COUNT(*) FROM documents 
            WHERE meta LIKE '%predicted_confidence%'
        """)
        confidence_count = cur.fetchone()[0]
        
        print(f"   Documents with predicted_label: {classified_count}/{doc_count}")
        print(f"   Documents with confidence scores: {confidence_count}/{doc_count}")
        
        if classified_count > 0:
            # Get unique labels
            cur.execute("""
                SELECT DISTINCT json_extract(meta, '$.predicted_label') as label
                FROM documents 
                WHERE json_extract(meta, '$.predicted_label') IS NOT NULL
            """)
            labels = [row[0] for row in cur.fetchall() if row[0]]
            if labels:
                print(f"   Unique labels found: {', '.join(labels)}")
        
        con.close()
        return True
        
    except sqlite3.Error as e:
        print(f"❌ SQLite error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("🗄️ SQLite Labels Quality Check")
    print("=" * 40)
    
    success = check_sqlite_labels()
    
    if success:
        print(f"\n✅ SQLite check completed")
    else:
        print(f"\n❌ SQLite check failed")

if __name__ == "__main__":
    main()
