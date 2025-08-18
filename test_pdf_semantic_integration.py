#!/usr/bin/env python3
"""
Test script to verify semantic chunking integration with PDF extractor.
"""

import tempfile
import os
from pathlib import Path

# Create a simple test PDF using reportlab if available
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

def create_test_pdf(pdf_path: Path) -> None:
    """Create a simple test PDF with multiple pages and sections."""
    if not REPORTLAB_AVAILABLE:
        print("ReportLab not available, skipping PDF creation")
        return
    
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    
    # Page 1 - Introduction
    c.drawString(100, height - 100, "1. INTRODUCTION")
    c.drawString(100, height - 140, "This is the introduction section of our test document.")
    c.drawString(100, height - 160, "It contains some basic information about the topic.")
    c.drawString(100, height - 180, "This section provides background and context.")
    c.drawString(100, height - 200, "We will discuss various aspects in the following sections.")
    
    c.showPage()
    
    # Page 2 - Methods
    c.drawString(100, height - 100, "2. METHODS")
    c.drawString(100, height - 140, "This section describes the methodology used in our study.")
    c.drawString(100, height - 160, "We employed various techniques and approaches.")
    c.drawString(100, height - 180, "The experimental setup was carefully designed.")
    c.drawString(100, height - 200, "Data collection followed standard protocols.")
    
    c.showPage()
    
    # Page 3 - Results
    c.drawString(100, height - 100, "3. RESULTS")
    c.drawString(100, height - 140, "Our findings show significant improvements.")
    c.drawString(100, height - 160, "The results exceeded our initial expectations.")
    c.drawString(100, height - 180, "Statistical analysis confirms the validity.")
    c.drawString(100, height - 200, "Performance metrics demonstrate effectiveness.")
    
    c.save()


def test_semantic_chunking_integration():
    """Test the complete semantic chunking integration."""
    
    # Test with simple text-based approach if no PDF creation available
    print("Testing semantic chunking integration with PDF extractor...")
    
    try:
        from bu_processor.bu_processor.pipeline.pdf_extractor import EnhancedPDFExtractor, ChunkingStrategy
        from bu_processor.bu_processor.config import settings
        
        # Create extractor
        extractor = EnhancedPDFExtractor()
        
        print(f"Semantic chunking enabled in config: {settings.ENABLE_SEMANTIC_CHUNKING}")
        
        # Test with a document if we can create one
        if REPORTLAB_AVAILABLE:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                pdf_path = Path(tmp_file.name)
                
            try:
                create_test_pdf(pdf_path)
                print(f"Created test PDF: {pdf_path}")
                
                # Test semantic chunking
                print("\nTesting SEMANTIC chunking strategy...")
                content = extractor.extract_text_from_pdf(
                    pdf_path, 
                    chunking_strategy=ChunkingStrategy.SEMANTIC,
                    max_chunk_size=200
                )
                
                print(f"Extraction successful!")
                print(f"- Text length: {len(content.text)}")
                print(f"- Page count: {content.page_count}")
                print(f"- Chunking enabled: {content.chunking_enabled}")
                print(f"- Number of chunks: {len(content.chunks)}")
                print(f"- Extraction method: {content.extraction_method}")
                print(f"- Pages available: {len(content.pages) if content.pages else 'None'}")
                
                if content.chunks:
                    print(f"\nFirst chunk details:")
                    chunk = content.chunks[0]
                    print(f"- Chunk ID: {chunk.chunk_id}")
                    print(f"- Text: {chunk.text[:100]}...")
                    print(f"- Page range: {chunk.page_start}-{chunk.page_end}")
                    print(f"- Section: {chunk.section}")
                    print(f"- Chunk type: {chunk.chunk_type}")
                    print(f"- Importance score: {chunk.importance_score}")
                    
                    if len(content.chunks) > 1:
                        print(f"\nLast chunk details:")
                        chunk = content.chunks[-1]
                        print(f"- Chunk ID: {chunk.chunk_id}")
                        print(f"- Text: {chunk.text[:100]}...")
                        print(f"- Page range: {chunk.page_start}-{chunk.page_end}")
                        print(f"- Section: {chunk.section}")
                
                # Test simple chunking for comparison
                print(f"\nTesting SIMPLE chunking strategy for comparison...")
                simple_content = extractor.extract_text_from_pdf(
                    pdf_path, 
                    chunking_strategy=ChunkingStrategy.SIMPLE,
                    max_chunk_size=200
                )
                
                print(f"Simple chunking:")
                print(f"- Number of chunks: {len(simple_content.chunks)}")
                print(f"- Chunking method: {simple_content.chunking_method}")
                
                print(f"\nIntegration test completed successfully!")
                
            finally:
                # Clean up
                if pdf_path.exists():
                    os.unlink(pdf_path)
                    
        else:
            print("ReportLab not available - cannot create test PDF")
            print("Testing configuration and imports only...")
            
            # Test that imports work
            from bu_processor.bu_processor.pipeline.chunk_entry import chunk_document_pages
            from bu_processor.bu_processor.semantic.greedy_boundary_chunker import GreedyBoundarySemanticChunker
            
            print("✓ Semantic chunking imports working")
            print("✓ Unified chunking entry point available")
            
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_semantic_chunking_integration()
    if success:
        print("\n🎉 Semantic chunking integration test PASSED!")
    else:
        print("\n❌ Semantic chunking integration test FAILED!")
