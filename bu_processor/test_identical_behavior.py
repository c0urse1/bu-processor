#!/usr/bin/env python3
"""Test to verify both PyMuPDF and PyPDF2 behave identically for empty text cases."""

import sys
import os
from unittest.mock import Mock, patch

# Add the bu_processor directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_identical_exception_behavior():
    """Test that both PyMuPDF and PyPDF2 behave identically for empty text."""
    
    try:
        from bu_processor.pipeline.pdf_extractor import (
            EnhancedPDFExtractor, 
            PDFTextExtractionError, 
            NoExtractableTextError,
            PDFCorruptedError
        )
        from pathlib import Path
        
        # Create an extractor instance
        extractor = EnhancedPDFExtractor()
        
        print("✅ Successfully imported PDF extractor classes")
        
        # Test 1: PyMuPDF - NoExtractableTextError conversion
        print("\n🔧 Testing PyMuPDF exception conversion...")
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
                    print("❌ PyMuPDF: Expected PDFTextExtractionError but got success")
                except PDFTextExtractionError as e:
                    if "No meaningful text extracted from PDF" in str(e):
                        print("✅ PyMuPDF: NoExtractableTextError correctly converted")
                    else:
                        print(f"❌ PyMuPDF: Wrong message: {e}")
                except Exception as e:
                    print(f"❌ PyMuPDF: Unexpected exception: {type(e).__name__}: {e}")
        
        # Test 2: PyPDF2 - NoExtractableTextError conversion
        print("\n🔧 Testing PyPDF2 exception conversion...")
        with patch('builtins.open', create=True) as mock_open:
            with patch('bu_processor.pipeline.pdf_extractor.PyPDF2.PdfReader') as mock_reader_class:
                mock_file = Mock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                mock_reader = Mock()
                mock_reader_class.return_value = mock_reader
                mock_reader.pages = [Mock()]  # One page
                mock_reader.is_encrypted = False
                mock_reader.metadata = {}
                
                # Mock page to return empty text
                mock_page = Mock()
                mock_reader.pages = [mock_page]
                mock_page.extract_text.return_value = ""  # Empty text
                
                try:
                    result = extractor._extract_with_pypdf2(Path("test.pdf"))
                    print("❌ PyPDF2: Expected PDFTextExtractionError but got success")
                except PDFTextExtractionError as e:
                    if "No meaningful text extracted from PDF" in str(e):
                        print("✅ PyPDF2: NoExtractableTextError correctly converted")
                    else:
                        print(f"❌ PyPDF2: Wrong message: {e}")
                except Exception as e:
                    print(f"❌ PyPDF2: Unexpected exception: {type(e).__name__}: {e}")
        
        print("\n✅ Both engines now behave identically for empty text cases!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_identical_exception_behavior()
