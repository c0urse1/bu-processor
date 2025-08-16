#!/usr/bin/env python3
"""Test to verify OCR functionality follows the same exception patterns."""

import sys
import os
from unittest.mock import Mock, patch

# Add the bu_processor directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_ocr_exception_behavior():
    """Test that OCR follows the same exception conversion pattern."""
    
    try:
        from bu_processor.pipeline.pdf_extractor import (
            EnhancedPDFExtractor, 
            PDFTextExtractionError, 
            NoExtractableTextError
        )
        from pathlib import Path
        
        # Create an extractor instance
        extractor = EnhancedPDFExtractor()
        
        print("✅ Successfully imported PDF extractor classes")
        
        # Test 1: OCR with no meaningful text should raise PDFTextExtractionError
        print("\n🔧 Testing OCR exception conversion...")
        
        # Mock OCR being available
        with patch('bu_processor.pipeline.pdf_extractor.OCR_AVAILABLE', True):
            with patch('bu_processor.pipeline.pdf_extractor.fitz.open') as mock_fitz_open:
                with patch('bu_processor.pipeline.pdf_extractor.Image') as mock_image:
                    with patch('bu_processor.pipeline.pdf_extractor.ocr') as mock_ocr:
                        
                        # Mock the PDF document
                        mock_doc = Mock()
                        mock_fitz_open.return_value.__enter__.return_value = mock_doc
                        mock_doc.__len__.return_value = 1  # One page
                        
                        # Mock page extraction
                        mock_page = Mock()
                        mock_doc.load_page.return_value = mock_page
                        mock_pix = Mock()
                        mock_page.get_pixmap.return_value = mock_pix
                        mock_pix.tobytes.return_value = b"fake_image_bytes"
                        
                        # Mock PIL Image
                        mock_img = Mock()
                        mock_image.open.return_value = mock_img
                        
                        # Mock OCR to return meaningless text (only symbols)
                        mock_ocr.image_to_string.return_value = "!!! @@@ ### $$$ %%%"  # No alphanumeric
                        
                        try:
                            result = extractor._extract_with_ocr(Path("test.pdf"))
                            print("❌ OCR: Expected PDFTextExtractionError but got success")
                        except PDFTextExtractionError as e:
                            if "No meaningful text extracted from PDF" in str(e):
                                print("✅ OCR: NoExtractableTextError correctly converted to PDFTextExtractionError")
                            else:
                                print(f"❌ OCR: Wrong message: {e}")
                        except Exception as e:
                            print(f"❌ OCR: Unexpected exception: {type(e).__name__}: {e}")
        
        # Test 2: Verify pytesseract is now importable for mocking
        print("\n🔧 Testing pytesseract import for test mocking...")
        try:
            import bu_processor.pipeline.pdf_extractor as pdf_extractor
            if hasattr(pdf_extractor, 'pytesseract'):
                print("✅ pytesseract is available at module level for test patching")
            else:
                print("❌ pytesseract not available at module level")
        except Exception as e:
            print(f"❌ Error checking pytesseract availability: {e}")
        
        print("\n✅ OCR exception behavior test completed!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_ocr_exception_behavior()
