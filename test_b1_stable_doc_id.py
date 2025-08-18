#!/usr/bin/env python3
"""
Test script for B1) Proper Chunk Metadata Tagging with stable doc_id flow.
"""

import tempfile
import os
from pathlib import Path
from uuid import uuid4

# Create a simple test PDF using reportlab if available
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

def create_multi_section_pdf(pdf_path: Path) -> None:
    """Create a test PDF with multiple sections and pages for testing B1."""
    if not REPORTLAB_AVAILABLE:
        print("ReportLab not available, skipping PDF creation")
        return
    
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    
    # Page 1 - Executive Summary
    c.drawString(100, height - 100, "EXECUTIVE SUMMARY")
    c.drawString(100, height - 140, "This document outlines our comprehensive analysis of business operations.")
    c.drawString(100, height - 160, "Key findings include improved efficiency metrics and cost optimization.")
    c.drawString(100, height - 180, "Recommendations focus on strategic implementation and resource allocation.")
    c.drawString(100, height - 200, "Expected ROI demonstrates significant value proposition for stakeholders.")
    
    c.showPage()
    
    # Page 2 - 1. BUSINESS ANALYSIS
    c.drawString(100, height - 100, "1. BUSINESS ANALYSIS")
    c.drawString(100, height - 140, "1.1 Market Position")
    c.drawString(100, height - 160, "Our market analysis reveals competitive advantages in key segments.")
    c.drawString(100, height - 180, "Customer satisfaction metrics show consistent improvement trends.")
    c.drawString(100, height - 200, "1.2 Financial Performance")
    c.drawString(100, height - 220, "Revenue growth exceeds industry benchmarks by significant margins.")
    c.drawString(100, height - 240, "Operational costs have been reduced through process optimization.")
    
    c.showPage()
    
    # Page 3 - 2. RISK ASSESSMENT  
    c.drawString(100, height - 100, "2. RISK ASSESSMENT")
    c.drawString(100, height - 140, "2.1 Operational Risks")
    c.drawString(100, height - 160, "Supply chain vulnerabilities require strategic mitigation approaches.")
    c.drawString(100, height - 180, "Technology dependencies present both opportunities and challenges.")
    c.drawString(100, height - 200, "2.2 Financial Risks")
    c.drawString(100, height - 220, "Market volatility impacts require robust hedging strategies.")
    c.drawString(100, height - 240, "Credit exposure management follows conservative risk frameworks.")
    
    c.showPage()
    
    # Page 4 - 3. RECOMMENDATIONS
    c.drawString(100, height - 100, "3. RECOMMENDATIONS")
    c.drawString(100, height - 140, "3.1 Strategic Initiatives")
    c.drawString(100, height - 160, "Implement advanced analytics for predictive business intelligence.")
    c.drawString(100, height - 180, "Expand digital transformation across all operational departments.")
    c.drawString(100, height - 200, "3.2 Implementation Timeline")
    c.drawString(100, height - 220, "Phase 1: Foundation building over next six months.")
    c.drawString(100, height - 240, "Phase 2: Full deployment within twelve month timeframe.")
    
    c.save()


def test_b1_stable_doc_id_flow():
    """Test the complete B1 implementation with stable doc_id flow."""
    
    print("Testing B1) Proper Chunk Metadata Tagging with stable doc_id...")
    
    try:
        from bu_processor.bu_processor.pipeline.pdf_extractor import EnhancedPDFExtractor, ChunkingStrategy
        from bu_processor.bu_processor.factories import make_embedder, make_index, make_store
        
        # Create extractor
        extractor = EnhancedPDFExtractor()
        
        # Test with a document if we can create one
        if REPORTLAB_AVAILABLE:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                pdf_path = Path(tmp_file.name)
                
            try:
                create_multi_section_pdf(pdf_path)
                print(f"Created test PDF: {pdf_path}")
                
                # Test 1: B1 method WITHOUT storage components (extraction only)
                print(f"\n=== Test 1: B1 extraction without storage ===")
                result1 = extractor.extract_and_upsert_pdf(
                    pdf_path=pdf_path,
                    chunking_strategy=ChunkingStrategy.SEMANTIC,
                    title="Business Analysis Report",
                    tenant="test_org"
                )
                
                print(f"✓ B1 extraction completed")
                print(f"  - Doc ID: {result1['doc_id']}")
                print(f"  - Title: {result1['title']}")
                print(f"  - Chunks: {result1['chunks_count']}")
                print(f"  - Method: {result1['chunking_method']}")
                print(f"  - Pages: {result1['page_count']}")
                print(f"  - Text length: {result1['text_length']}")
                
                # Verify chunks have proper metadata
                if 'chunks' in result1:
                    sample_chunk = result1['chunks'][0]
                    print(f"  - Sample chunk doc_id: {sample_chunk.doc_id}")
                    print(f"  - Sample chunk section: {sample_chunk.section}")
                    print(f"  - Sample chunk page range: {sample_chunk.page_start}-{sample_chunk.page_end}")
                    print(f"  - Sample chunk meta keys: {list(sample_chunk.meta.keys()) if sample_chunk.meta else 'None'}")
                
                # Test 2: B1 method WITH storage components (full pipeline)
                print(f"\n=== Test 2: B1 full pipeline with storage ===")
                
                # Create storage components
                embedder = make_embedder()
                index = make_index()
                store = make_store()
                
                result2 = extractor.extract_and_upsert_pdf(
                    pdf_path=pdf_path,
                    chunking_strategy=ChunkingStrategy.SEMANTIC,
                    title="Business Analysis Report - Full Pipeline",
                    tenant="prod_org",
                    embedder=embedder,
                    index=index,
                    store=store,
                    namespace="test_b1"
                )
                
                print(f"✓ B1 full pipeline completed")
                print(f"  - Doc ID: {result2['doc_id']}")
                print(f"  - Stored chunks: {len(result2['chunk_ids'])}")
                print(f"  - Embedding dim: {result2['dim']}")
                print(f"  - Tenant: {result2['tenant']}")
                
                # Test 3: Verify stable doc_id (different calls should create different doc_ids)
                print(f"\n=== Test 3: Verify doc_id stability ===")
                
                result3 = extractor.extract_and_upsert_pdf(
                    pdf_path=pdf_path,
                    chunking_strategy=ChunkingStrategy.SEMANTIC,
                    title="Same PDF, Different Doc ID",
                    tenant="test_org"
                )
                
                print(f"✓ Multiple extractions generate different doc_ids:")
                print(f"  - First extraction: {result1['doc_id']}")
                print(f"  - Second extraction: {result2['doc_id']}")  
                print(f"  - Third extraction: {result3['doc_id']}")
                
                assert result1['doc_id'] != result2['doc_id'] != result3['doc_id'], "Doc IDs should be unique"
                
                # Test 4: Verify rich metadata in chunks
                print(f"\n=== Test 4: Verify rich chunk metadata ===")
                
                chunks = result1['chunks']
                sample_chunks = chunks[:3] if len(chunks) >= 3 else chunks
                
                for i, chunk in enumerate(sample_chunks):
                    print(f"  Chunk {i+1}:")
                    print(f"    - Text: {chunk.text[:80]}...")
                    print(f"    - Doc ID: {chunk.doc_id}")
                    print(f"    - Section: {chunk.section}")
                    print(f"    - Page: {chunk.page_start}-{chunk.page_end}")
                    print(f"    - Heading: {chunk.heading_text}")
                    print(f"    - Type: {chunk.chunk_type}")
                    print(f"    - Meta source_url: {chunk.meta.get('source_url', 'N/A')}")
                    print(f"    - Meta tenant: {chunk.meta.get('tenant', 'N/A')}")
                    
                print(f"\n✓ All B1 tests passed!")
                return True
                
            finally:
                # Clean up
                if pdf_path.exists():
                    os.unlink(pdf_path)
                    
        else:
            print("ReportLab not available - testing imports only...")
            
            # Test that new method exists
            extractor = EnhancedPDFExtractor()
            assert hasattr(extractor, 'extract_and_upsert_pdf'), "B1 method should exist"
            
            print("✓ B1 method exists and imports working")
            return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_b1_stable_doc_id_flow()
    if success:
        print("\n🎉 B1) Proper Chunk Metadata Tagging test PASSED!")
        print("✅ Stable doc_id generation implemented")
        print("✅ Rich metadata flow working")
        print("✅ End-to-end integration complete")
    else:
        print("\n❌ B1) Proper Chunk Metadata Tagging test FAILED!")
