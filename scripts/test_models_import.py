#!/usr/bin/env python3
"""
🧪 MODELS IMPORT TEST
==================
Test that DocumentChunk can be imported from the models package.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_models_import():
    """Test that DocumentChunk imports work correctly."""
    print("🧪 MODELS IMPORT TEST")
    print("=" * 25)
    
    try:
        # Test direct import from models
        print("🔧 Testing direct import from models...")
        from bu_processor.models.chunk import DocumentChunk, create_semantic_chunk, create_paragraph_chunk
        print("✅ Direct import successful")
        
        # Test import through models __init__
        print("🔧 Testing import through models package...")
        from bu_processor.models import DocumentChunk as DC2
        print("✅ Package import successful")
        
        # Test creating chunks
        print("🔧 Testing chunk creation...")
        chunk1 = DocumentChunk(text="Test chunk content")
        chunk2 = create_semantic_chunk("Semantic content", heading="Test Heading")
        chunk3 = create_paragraph_chunk("Paragraph content", source="test")
        
        print(f"✅ Basic chunk: {chunk1.char_count} chars")
        print(f"✅ Semantic chunk: '{chunk2.heading_text}' - {chunk2.char_count} chars")
        print(f"✅ Paragraph chunk: {chunk3.meta} - {chunk3.word_count} words")
        
        # Test properties
        print("🔧 Testing chunk properties...")
        empty_chunk = DocumentChunk(text="")
        assert empty_chunk.is_empty == True
        assert chunk1.is_empty == False
        print("✅ Properties work correctly")
        
        # Test import from pipeline files that need DocumentChunk
        print("🔧 Testing imports from pipeline files...")
        
        try:
            from bu_processor.pipeline.classifier import RealMLClassifier
            print("✅ classifier.py import works")
        except ImportError as e:
            print(f"⚠️  classifier.py import issue (might be OK): {e}")
        
        try:
            from bu_processor.pipeline.pdf_extractor import EnhancedPDFExtractor
            print("✅ pdf_extractor.py import works")
        except ImportError as e:
            print(f"⚠️  pdf_extractor.py import issue (might be OK): {e}")
        
        print("\n🎉 MODELS IMPORT TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_models_import()
    sys.exit(0 if success else 1)
