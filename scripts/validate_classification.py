#!/usr/bin/env python3
"""
📊 ROBUST & FIELD-TESTED CLASSIFICATION VALIDATION SCRIPT
=========================================================

Validates classification results by processing PDFs in batches and generating
a comprehensive CSV report with classification results.

Features:
- Bearer token authentication via BU_API_TOKEN
- IPv4 API connection (127.0.0.1)
- Per-chunk classification for better accuracy on long PDFs
- Semantic chunking strategy
- Robust field mappings with fallbacks
- Clean error handling
"""

import csv
import os
import time
import requests
from pathlib import Path

# Configuration - IPv4 enforced, robust defaults
API_BASE = os.getenv("BU_API_BASE", "http://127.0.0.1:8000")
API = f"{API_BASE}/classify/pdf"
API_TOKEN = os.getenv("BU_API_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

FOLDER = Path("data/pdfs")
OUT_CSV = Path("out/classification_report.csv")

# Ensure directories exist
FOLDER.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def wait_for_api(timeout=60):
    """Wait for API to be available via /docs endpoint."""
    print(f"🔍 Checking API availability at {API_BASE}...")
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{API_BASE}/docs", timeout=3)
            if r.status_code == 200:
                print(f"✅ API is available")
                return True
        except requests.RequestException:
            pass
        print(f"⏳ Waiting... ({int(time.time() - start)}s)")
        time.sleep(1)
    
    raise RuntimeError(f"❌ API nicht erreichbar unter {API_BASE}")

def classify_pdf(path: Path, chunking="semantic", max_chunk_size=1000, per_chunk=True):
    """
    Classify PDF with robust defaults for field use.
    
    - chunking="semantic": Better content understanding
    - per_chunk=True: More stable signals for long PDFs
    - max_chunk_size=1000: Good balance of context vs. processing
    """
    with path.open("rb") as f:
        files = {"file": (path.name, f, "application/pdf")}
        params = {
            "chunking_strategy": chunking,
            "max_chunk_size": max_chunk_size,
            "classify_chunks_individually": str(per_chunk).lower(),
        }
        return requests.post(API, files=files, params=params,
                           headers=HEADERS, timeout=600)

def main():
    """Main function with field-tested robustness."""
    print(f"🔑 Using token: {'✅ Set' if API_TOKEN else '❌ Not set'}")
    print(f"📡 API endpoint: {API}")
    print(f"📁 PDF folder: {FOLDER}")
    print(f"📄 Output CSV: {OUT_CSV}")
    
    # Step 1: Wait for API
    wait_for_api()
    
    # Step 2: Find PDFs
    pdfs = sorted(p for p in FOLDER.glob("*.pdf"))
    if not pdfs:
        print(f"❌ Keine PDFs in {FOLDER}")
        return
    
    print(f"� Found {len(pdfs)} PDF files to process")
    
    # Step 3: Process with robust field mapping
    fields = ["filename", "status", "category", "category_label", "confidence", 
              "is_confident", "pages", "text_length", "error"]
    
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        
        for pdf in pdfs:
            print(f"[INFO] Klassifiziere: {pdf.name}")
            try:
                # Use semantic chunking + per-chunk classification
                r = classify_pdf(pdf)  # semantic + per_chunk=True
                
                row = dict.fromkeys(fields, "")
                row["filename"] = pdf.name
                row["status"] = r.status_code
                
                # Parse JSON with fallback
                data = {}
                if r.headers.get("content-type", "").startswith("application/json"):
                    data = r.json()
                
                if r.status_code == 200:
                    # Success - extract with fallbacks
                    row["category"] = data.get("category")
                    row["category_label"] = data.get("category_label") or data.get("label")
                    row["confidence"] = data.get("confidence")
                    row["is_confident"] = data.get("is_confident")
                    # Robust page count mapping
                    row["pages"] = data.get("pages") or data.get("page_count")
                    # Robust text length mapping  
                    row["text_length"] = data.get("text_length") or data.get("total_text_length")
                    
                    print(f"   ✅ {row['category_label']} (confidence: {row['confidence']}, pages: {row['pages']})")
                else:
                    # Error handling
                    row["error"] = str(data.get("detail") or r.text)
                    print(f"   ❌ Error {r.status_code}: {row['error']}")
                
                w.writerow(row)
                
            except requests.RequestException as e:
                print(f"[ERR] HTTP: {e}")
                raise  # Re-raise for visibility
            
            # Short delay to be API-friendly
            time.sleep(0.3)
    
    print(f"[OK] Report: {OUT_CSV}")

if __name__ == "__main__":
    main()

def test_api_connection():
    """Test if the API is available and responding"""
    try:
        print("🔍 Testing API connection...")
        response = requests.get(f"{API_BASE}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is available!")
            print(f"   Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print(f"   Make sure the API server is running on {API_BASE}")
        return False

def main():
    """Main validation function"""
    print("📊 PDF Classification Validation Script")
    print("=" * 50)
    
    # Test API connection first
    if not test_api_connection():
        print("\n❌ Cannot proceed without API connection")
        sys.exit(1)
    
    # Find PDF files
    pdfs = sorted(p for p in FOLDER.glob("*.pdf"))
    
    if not pdfs:
        print(f"\n❌ No PDF files found in {FOLDER}")
        print(f"   Please add PDF files to {FOLDER.absolute()}")
        return
    
    print(f"\n📄 Found {len(pdfs)} PDF files to process")
    print(f"📝 Output will be written to: {OUT_CSV.absolute()}")
    
    # CSV field definitions
    fields = [
        "filename", "status", "category", "category_label", "confidence", 
        "is_confident", "pages", "text_length", "processing_time", 
        "file_size", "extraction_method", "error"
    ]
    
    # Process all PDFs
    try:
        with OUT_CSV.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            
            for i, pdf in enumerate(pdfs, 1):
                print(f"\n[{i}/{len(pdfs)}] Processing: {pdf.name}")
                
                # Get file size
                file_size = pdf.stat().st_size
                print(f"   File size: {file_size:,} bytes")
                
                # Classify the PDF
                start_time = time.time()
                result = classify_pdf(
                    pdf, 
                    chunking="simple", 
                    max_chunk_size=1000, 
                    per_chunk=False
                )
                processing_time = time.time() - start_time
                
                # Prepare CSV row
                row = {field: "" for field in fields}
                row["filename"] = pdf.name
                row["status"] = result["status"]
                row["file_size"] = file_size
                row["processing_time"] = f"{processing_time:.2f}"
                
                if result["status"] == 200:
                    # Successful classification
                    data = result["data"]
                    row["category"] = data.get("category", "")
                    row["category_label"] = data.get("category_label") or data.get("label", "")
                    row["confidence"] = data.get("confidence", "")
                    row["is_confident"] = data.get("is_confident", "")
                    row["pages"] = data.get("page_count", "")
                    row["text_length"] = data.get("text_length", "")
                    row["extraction_method"] = data.get("extraction_method", "")
                    
                    print(f"   ✅ Classification: {row['category_label']} (confidence: {row['confidence']})")
                    
                else:
                    # Error occurred
                    if isinstance(result.get("data"), dict):
                        error_detail = result["data"].get("detail") or result["data"].get("error", "")
                    else:
                        error_detail = result.get("error") or result.get("raw", "Unknown error")
                    
                    row["error"] = str(error_detail)
                    print(f"   ❌ Error: {row['error']}")
                
                # Write row to CSV
                writer.writerow(row)
                
                # Small delay to be nice to the API
                time.sleep(0.5)
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Processing interrupted by user")
        print(f"   Partial results saved to: {OUT_CSV}")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        raise
    
    print(f"\n✅ Validation completed!")
    print(f"📊 Full report saved to: {OUT_CSV.absolute()}")
    
    # Show summary statistics
    show_summary_stats()

def show_summary_stats():
    """Show summary statistics from the generated report"""
    try:
        if not OUT_CSV.exists():
            return
        
        print(f"\n📈 Summary Statistics")
        print("-" * 30)
        
        with OUT_CSV.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            rows = list(reader)
        
        total = len(rows)
        successful = sum(1 for row in rows if row["status"] == "200")
        failed = total - successful
        
        print(f"Total PDFs processed: {total}")
        print(f"Successful classifications: {successful}")
        print(f"Failed classifications: {failed}")
        
        if successful > 0:
            # Calculate confidence statistics
            confidences = [float(row["confidence"]) for row in rows if row["confidence"]]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                print(f"Average confidence: {avg_confidence:.3f}")
            
            # Show category distribution
            categories = {}
            for row in rows:
                if row["category_label"]:
                    categories[row["category_label"]] = categories.get(row["category_label"], 0) + 1
            
            if categories:
                print(f"\nCategory distribution:")
                for category, count in sorted(categories.items()):
                    print(f"  {category}: {count}")
        
    except Exception as e:
        print(f"Error generating summary: {e}")

if __name__ == "__main__":
    main()
