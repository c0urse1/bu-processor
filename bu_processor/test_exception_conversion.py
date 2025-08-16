#!/usr/bin/env python3
"""Quick test to verify the exception handling in _extract_with_pymupdf."""

import sys
import os
from unittest.mock import Mock, patch

# Add the bu_processor directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_exception_conversion():
    """Test that NoExtractableTextError is properly converted to PDFTextExtractionError."""
    
    try:
        from bu_processor.pipeline.pdf_extractor import (
            EnhancedPDFExtractor, 
            PDFTextExtractionError, 
            NoExtractableTextError,
            PDFCorruptedError,
            PDFTooLargeError
        )
        
        # Create an extractor instance
        extractor = EnhancedPDFExtractor()
        
        print("✅ Successfully imported PDF extractor classes")
        
        # Test 1: NoExtractableTextError should be converted to PDFTextExtractionError
        with patch('bu_processor.pipeline.pdf_extractor.fitz.open') as mock_fitz_open:
            mock_doc = Mock()
            mock_fitz_open.return_value = mock_doc
            mock_doc.needs_pass = False
            mock_doc.page_count = 1
            mock_doc.metadata = {}
            mock_doc.is_pdf = True
            
            # Mock _extract_text_from_fitz_doc to raise NoExtractableTextError
            with patch.object(extractor, '_extract_text_from_fitz_doc') as mock_extract:
                mock_extract.side_effect = NoExtractableTextError("no text or not meaningful")
                
                try:
                    from pathlib import Path
                    result = extractor._extract_with_pymupdf(Path("test.pdf"))
                    print("❌ Expected PDFTextExtractionError but got success")
                except PDFTextExtractionError as e:
                    if "No meaningful text extracted from PDF" in str(e):
                        print("✅ NoExtractableTextError correctly converted to PDFTextExtractionError")
                    else:
                        print(f"❌ Wrong message in PDFTextExtractionError: {e}")
                except Exception as e:
                    print(f"❌ Unexpected exception type: {type(e).__name__}: {e}")
        
        print("✅ Exception conversion test completed")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_exception_conversion()
