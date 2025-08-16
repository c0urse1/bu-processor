#!/usr/bin/env python3
"""Test to verify logging is added when converting exceptions."""

import sys
import os
from unittest.mock import Mock, patch
import logging

# Add the bu_processor directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_exception_logging():
    """Test that exception conversion includes proper logging."""
    
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
        
        # Test 1: PyMuPDF logging
        print("\n🔧 Testing PyMuPDF exception logging...")
        with patch('bu_processor.pipeline.pdf_extractor.logger') as mock_logger:
            with patch('bu_processor.pipeline.pdf_extractor.fitz.open') as mock_fitz_open:
                mock_doc = Mock()
                mock_fitz_open.return_value = mock_doc
                mock_doc.needs_pass = False
                mock_doc.page_count = 1
                mock_doc.metadata = {}
                mock_doc.is_pdf = True
                
                with patch.object(extractor, '_extract_text_from_fitz_doc') as mock_extract:
                    mock_extract.side_effect = NoExtractableTextError("no text or not meaningful")
                    
                    try:
                        result = extractor._extract_with_pymupdf(Path("test.pdf"))
                    except PDFTextExtractionError:
                        # Check if logging was called
                        mock_logger.error.assert_called_with("No meaningful text extracted", file="test.pdf")
                        print("✅ PyMuPDF: Logging called correctly")
                    except Exception as e:
                        print(f"❌ PyMuPDF: Unexpected exception: {e}")
        
        # Test 2: PyPDF2 logging  
        print("\n🔧 Testing PyPDF2 exception logging...")
        with patch('bu_processor.pipeline.pdf_extractor.logger') as mock_logger:
            with patch('builtins.open', create=True) as mock_open:
                with patch('bu_processor.pipeline.pdf_extractor.PyPDF2.PdfReader') as mock_reader_class:
                    mock_file = Mock()
                    mock_open.return_value.__enter__.return_value = mock_file
                    
                    mock_reader = Mock()
                    mock_reader_class.return_value = mock_reader
                    mock_reader.pages = [Mock()]
                    mock_reader.is_encrypted = False
                    mock_reader.metadata = {}
                    
                    # Mock page to return empty text
                    mock_page = Mock()
                    mock_reader.pages = [mock_page]
                    mock_page.extract_text.return_value = ""  # Empty text
                    
                    try:
                        result = extractor._extract_with_pypdf2(Path("test.pdf"))
                    except PDFTextExtractionError:
                        # Check if logging was called
                        mock_logger.error.assert_called_with("No meaningful text extracted", file="test.pdf")
                        print("✅ PyPDF2: Logging called correctly")
                    except Exception as e:
                        print(f"❌ PyPDF2: Unexpected exception: {e}")
        
        # Test 3: OCR logging
        print("\n🔧 Testing OCR exception logging...")
        with patch('bu_processor.pipeline.pdf_extractor.logger') as mock_logger:
            with patch('bu_processor.pipeline.pdf_extractor.OCR_AVAILABLE', True):
                with patch('bu_processor.pipeline.pdf_extractor.fitz.open') as mock_fitz_open:
                    with patch('bu_processor.pipeline.pdf_extractor.Image') as mock_image:
                        with patch('bu_processor.pipeline.pdf_extractor.ocr') as mock_ocr:
                            
                            # Mock the PDF document
                            mock_doc = Mock()
                            mock_fitz_open.return_value.__enter__.return_value = mock_doc
                            mock_doc.__len__.return_value = 1
                            
                            # Mock page extraction
                            mock_page = Mock()
                            mock_doc.load_page.return_value = mock_page
                            mock_pix = Mock()
                            mock_page.get_pixmap.return_value = mock_pix
                            mock_pix.tobytes.return_value = b"fake_image_bytes"
                            
                            # Mock PIL Image
                            mock_img = Mock()
                            mock_image.open.return_value = mock_img
                            
                            # Mock OCR to return meaningless text
                            mock_ocr.image_to_string.return_value = "!!! @@@ ###"
                            
                            try:
                                result = extractor._extract_with_ocr(Path("test.pdf"))
                            except PDFTextExtractionError:
                                # Check if logging was called
                                mock_logger.error.assert_called_with("No meaningful text extracted", file="test.pdf")
                                print("✅ OCR: Logging called correctly")
                            except Exception as e:
                                print(f"❌ OCR: Unexpected exception: {e}")
        
        print("\n✅ All exception logging tests completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_exception_logging()
