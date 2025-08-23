#!/usr/bin/env python3
"""
🚀 BULK PDF INGESTION SCRIPT
===========================

Robust batch processing for massive PDF ingestion with:
- Hash-based deduplication to avoid re-processing
- Resume capability with state tracking
- Batch processing with proper backoff
- Integration with existing project factories
- SQLite + Pinecone integration
"""

import os
import json
import hashlib
import time
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import BU-Processor factories and utilities
sys.path.append(str(Path(__file__).parent.parent))

try:
    from bu_processor.factories import make_embedder, make_index, make_store
    from bu_processor.pipeline.upsert_pipeline import embed_and_index_chunks
    from bu_processor.pipeline.classifier import extract_text_from_pdf
    from bu_processor.ml.classifier import RealMLClassifier
    from bu_processor.core.config import get_config
    from bu_processor.pipeline.pdf_extractor import ChunkingStrategy
    print("✅ Successfully imported BU-Processor modules")
except ImportError as e:
    print(f"❌ Failed to import BU-Processor modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configuration
PDF_DIR = Path("data/pdfs")                    # Input PDFs directory
DB_PATH = Path("bu_store.db")                  # SQLite database path
STATE_FILE = Path("out/ingest_state.jsonl")   # Progress tracking file
BATCH_SIZE = 10                                # Process PDFs in batches
BACKOFF_TIME = 0.5                            # Delay between operations

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    """Calculate SHA256 hash of a file for deduplication"""
    hash_sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def already_ingested(conn: sqlite3.Connection, file_hash: str) -> bool:
    """Check if a file with this hash has already been ingested"""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM documents WHERE meta LIKE ?", 
        (f'%"file_hash": "{file_hash}"%',)
    )
    return cursor.fetchone() is not None

def write_state(record: Dict[str, Any]):
    """Write processing state to JSONL file for resume capability"""
    with STATE_FILE.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

def load_existing_state() -> Dict[str, Dict[str, Any]]:
    """Load existing processing state from JSONL file"""
    state = {}
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open("r", encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        record = json.loads(line.strip())
                        if "file" in record:
                            state[record["file"]] = record
        except Exception as e:
            print(f"⚠️  Warning: Could not load state file: {e}")
    return state

def process_pdf_batch(
    pdf_batch: List[Path], 
    embedder, 
    index, 
    store, 
    conn: sqlite3.Connection,
    existing_state: Dict[str, Dict[str, Any]]
) -> Dict[str, int]:
    """Process a batch of PDF files"""
    
    stats = {"processed": 0, "skipped": 0, "errors": 0, "uploaded_vectors": 0}
    
    for pdf_path in pdf_batch:
        print(f"\n📄 Processing: {pdf_path.name}")
        
        # Check if already processed successfully
        if pdf_path.name in existing_state:
            last_state = existing_state[pdf_path.name]
            if last_state.get("status") == "ok":
                print(f"   ✅ Already processed successfully - skipping")
                stats["skipped"] += 1
                continue
        
        try:
            # Calculate file hash for deduplication
            file_hash = sha256_file(pdf_path)
            print(f"   🔍 File hash: {file_hash[:12]}...")
            
            # Check if already in database
            if already_ingested(conn, file_hash):
                print(f"   ⚠️  Already ingested in database - skipping")
                write_state({
                    "file": pdf_path.name,
                    "status": "skipped",
                    "reason": "already_ingested",
                    "file_hash": file_hash,
                    "timestamp": time.time()
                })
                stats["skipped"] += 1
                continue
            
            # Extract text from PDF  
            print(f"   📝 Extracting text from PDF...")
            extracted_content = extract_text_from_pdf(
                str(pdf_path),
                chunking_strategy=ChunkingStrategy.SEMANTIC,
                max_chunk_size=1000,
                enable_chunking=True
            )
            
            if not extracted_content.text or len(extracted_content.text.strip()) < 10:
                raise ValueError("No text extracted from PDF or text too short")
            
            print(f"   📦 Extracted text length: {len(extracted_content.text)}")
            
            # Convert to chunks format for embed_and_index_chunks
            if extracted_content.chunks:
                # Use existing chunks from extraction
                chunks = [
                    {
                        "text": chunk.text,
                        "page": chunk.page_start if hasattr(chunk, 'page_start') else None,
                        "section": chunk.heading_text if hasattr(chunk, 'heading_text') else None,
                        "meta": {
                            "chunk_id": getattr(chunk, 'chunk_id', None),
                            "chunk_type": getattr(chunk, 'chunk_type', 'text'),
                            "importance_score": getattr(chunk, 'importance_score', None)
                        }
                    }
                    for chunk in extracted_content.chunks
                ]
                print(f"   📦 Using {len(chunks)} chunks from extraction")
            else:
                # Create simple chunks from text
                chunk_size = 1000
                text = extracted_content.text
                chunks = [
                    {
                        "text": text[i:i + chunk_size],
                        "page": None,
                        "section": None,
                        "meta": {"chunk_index": i // chunk_size}
                    }
                    for i in range(0, len(text), chunk_size)
                ]
                print(f"   📦 Created {len(chunks)} simple chunks")
            
            # Base metadata from extraction
            metadata = extracted_content.metadata or {}
            
            # Enrich metadata
            enhanced_metadata = metadata.copy()
            enhanced_metadata.update({
                "file_name": pdf_path.name,
                "file_hash": file_hash,
                "source": "BU-Document",  # Business document source
                "ingestion_timestamp": time.time(),
                "ingestion_method": "bulk_ingestion",
                "chunk_count": len(chunks),
                "extraction_method": extracted_content.extraction_method,
                "page_count": extracted_content.page_count,
                "text_length": len(extracted_content.text)
            })
            
            # 🤖 CLASSIFICATION: Add predicted category to metadata
            print(f"   🔍 Classifying document...")
            try:
                clf = RealMLClassifier()
                cls = clf.classify_pdf(
                    str(pdf_path),
                    chunking_strategy="semantic",
                    max_chunk_size=1000,
                    classify_chunks_individually=True
                )
                
                if isinstance(cls, dict) and cls.get("error"):
                    raise RuntimeError(f"Klassifikation fehlgeschlagen: {cls['error']}")
                
                # Normalize classification result to dict
                d = cls if isinstance(cls, dict) else getattr(cls, "model_dump", lambda: {})()
                if not d: 
                    d = getattr(cls, "__dict__", {})
                
                # Add classification results to metadata
                enhanced_metadata["predicted_category"] = d.get("category")
                enhanced_metadata["predicted_label"] = d.get("category_label") or d.get("label")
                enhanced_metadata["predicted_confidence"] = d.get("confidence")
                enhanced_metadata["classification_timestamp"] = time.time()
                
                # Log classification result
                category_label = enhanced_metadata["predicted_label"] or "Unknown"
                confidence = enhanced_metadata["predicted_confidence"] or 0.0
                print(f"   📊 Classification: {category_label} (confidence: {confidence:.3f})")
                
            except Exception as e:
                print(f"   ⚠️  Classification failed: {e}")
                # Continue with ingestion even if classification fails
                enhanced_metadata["predicted_category"] = None
                enhanced_metadata["predicted_label"] = "Classification_Failed"
                enhanced_metadata["predicted_confidence"] = 0.0
                enhanced_metadata["classification_error"] = str(e)
            
            # Embed and index chunks with enriched metadata
            print(f"   🧠 Generating embeddings and indexing...")
            result = embed_and_index_chunks(
                chunks=chunks,
                embedder=embedder,
                index=index,
                store=store,
                doc_meta=enhanced_metadata
            )
            
            # Log success
            uploaded_vectors = result.get("uploaded", 0)
            stored_chunks = result.get("stored_chunks", 0)
            
            print(f"   ✅ Success: {uploaded_vectors} vectors uploaded, {stored_chunks} chunks stored")
            
            write_state({
                "file": pdf_path.name,
                "status": "ok",
                "uploaded_vectors": uploaded_vectors,
                "stored_chunks": stored_chunks,
                "chunk_count": len(chunks),
                "file_hash": file_hash,
                "timestamp": time.time()
            })
            
            stats["processed"] += 1
            stats["uploaded_vectors"] += uploaded_vectors
            
            # Small delay to be nice to Pinecone
            time.sleep(BACKOFF_TIME)
            
        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}")
            
            write_state({
                "file": pdf_path.name,
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            })
            
            stats["errors"] += 1
            
            # Longer delay after errors
            time.sleep(1.0)
    
    return stats

def main():
    """Main bulk ingestion function"""
    print("🚀 Bulk PDF Ingestion Script")
    print("=" * 50)
    
    # Find PDF files
    pdf_files = sorted(p for p in PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {PDF_DIR}")
        print(f"   Please add PDF files to {PDF_DIR.absolute()}")
        sys.exit(0)
    
    print(f"📁 Found {len(pdf_files)} PDF files to process")
    
    # Load existing state for resume capability
    existing_state = load_existing_state()
    if existing_state:
        print(f"📋 Loaded state for {len(existing_state)} previously processed files")
    
    # Initialize components
    print(f"\n🔧 Initializing embedder, index, and store...")
    
    try:
        config = get_config()
        embedder = make_embedder()
        index = make_index()  # Expects Pinecone config from environment
        store = make_store()  # SQLite store
        
        print(f"   ✅ Embedder: {type(embedder).__name__}")
        print(f"   ✅ Index: {type(index).__name__}")
        print(f"   ✅ Store: {type(store).__name__}")
        
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        print("   Make sure your .env file is configured with required API keys")
        sys.exit(1)
    
    # Connect to SQLite database
    conn = sqlite3.connect(DB_PATH)
    
    # Process PDFs in batches
    total_stats = {"processed": 0, "skipped": 0, "errors": 0, "uploaded_vectors": 0}
    
    try:
        # Split into batches
        batches = [pdf_files[i:i + BATCH_SIZE] for i in range(0, len(pdf_files), BATCH_SIZE)]
        
        print(f"\n📦 Processing {len(pdf_files)} files in {len(batches)} batches")
        
        for batch_num, batch in enumerate(batches, 1):
            print(f"\n🔄 Processing batch {batch_num}/{len(batches)} ({len(batch)} files)")
            
            batch_stats = process_pdf_batch(
                batch, embedder, index, store, conn, existing_state
            )
            
            # Update total stats
            for key in total_stats:
                total_stats[key] += batch_stats[key]
            
            # Show batch results
            print(f"\n📊 Batch {batch_num} Results:")
            print(f"   Processed: {batch_stats['processed']}")
            print(f"   Skipped: {batch_stats['skipped']}")
            print(f"   Errors: {batch_stats['errors']}")
            print(f"   Vectors uploaded: {batch_stats['uploaded_vectors']}")
            
            # Delay between batches
            if batch_num < len(batches):
                print(f"   ⏳ Waiting {BACKOFF_TIME}s before next batch...")
                time.sleep(BACKOFF_TIME)
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Processing interrupted by user")
        print(f"   Progress has been saved to {STATE_FILE}")
        print(f"   You can resume processing by running the script again")
    
    finally:
        conn.close()
    
    # Final summary
    print(f"\n🎉 Bulk ingestion completed!")
    print(f"📊 Final Statistics:")
    print(f"   Total files processed: {total_stats['processed']}")
    print(f"   Total files skipped: {total_stats['skipped']}")
    print(f"   Total errors: {total_stats['errors']}")
    print(f"   Total vectors uploaded: {total_stats['uploaded_vectors']}")
    print(f"   State file: {STATE_FILE.absolute()}")
    
    if total_stats['errors'] > 0:
        print(f"\n⚠️  {total_stats['errors']} files had errors. Check the state file for details.")
    
    if total_stats['processed'] > 0:
        print(f"\n✅ Successfully processed {total_stats['processed']} files!")

if __name__ == "__main__":
    main()
