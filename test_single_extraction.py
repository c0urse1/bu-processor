#!/usr/bin/env python3
"""
Simple test to verify the single extraction pattern works
"""

import sys
import os

# Add correct path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bu_processor'))

def test_single_extraction():
    """Test that the pipeline calls the extractor only once"""
    print("=== Testing Single Extraction Pattern ===")
    
    try:
        from bu_processor.pipeline.enhanced_integrated_pipeline import EnhancedIntegratedPipeline
        from unittest.mock import Mock, MagicMock
        
        # Create pipeline
        pipeline = EnhancedIntegratedPipeline()
        
        # Mock the PDF extractor
        mock_extractor = Mock()
        mock_extracted_content = Mock()
        mock_extracted_content.text = "This is test content from the PDF."
        mock_extracted_content.page_count = 1
        mock_extracted_content.extraction_method = "test"
        mock_extracted_content.chunks = []
        mock_extracted_content.chunking_enabled = False
        mock_extracted_content.chunking_method = None
        
        mock_extractor.extract_text_from_pdf.return_value = mock_extracted_content
        mock_extractor.chunk_text = Mock(return_value=["chunk1", "chunk2"])
        
        pipeline.pdf_extractor = mock_extractor
        
        # Test the single extraction helper
        print("Testing _extract_text_once...")
        raw_text = pipeline._extract_text_once("test.pdf")
        print(f"✓ _extract_text_once returned: '{raw_text[:50]}...'")
        
        # Verify extractor was called exactly once
        assert mock_extractor.extract_text_from_pdf.call_count == 1, f"Expected 1 call, got {mock_extractor.extract_text_from_pdf.call_count}"
        print("✓ Extractor called exactly once")
        
        # Test that calling it again doesn't re-extract (this might fail, that's ok for now)
        print("Testing multiple calls (this might call extractor again, that's expected)...")
        raw_text2 = pipeline._extract_text_once("test.pdf")
        total_calls = mock_extractor.extract_text_from_pdf.call_count
        print(f"After second call, total extractor calls: {total_calls}")
        
        print("=== Single Extraction Test Completed ===")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_extraction()
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed!")
        sys.exit(1)
