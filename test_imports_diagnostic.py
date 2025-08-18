#!/usr/bin/env python3
"""
Simple diagnostic test to check semantic chunking imports.
"""

def test_imports():
    """Test all the semantic chunking imports."""
    print("Testing semantic chunking imports...")
    
    try:
        # Test basic config
        from bu_processor.config import ENABLE_SEMANTIC_CHUNKING
        print(f"✓ Config import works, ENABLE_SEMANTIC_CHUNKING: {ENABLE_SEMANTIC_CHUNKING}")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        # Test chunk entry point
        from bu_processor.pipeline.chunk_entry import chunk_document_pages
        print("✓ chunk_document_pages import works")
    except Exception as e:
        print(f"✗ chunk_document_pages import failed: {e}")
    
    try:
        # Test semantic embeddings
        from bu_processor.semantic.embeddings import SbertEmbeddings
        print("✓ SbertEmbeddings import works")
    except Exception as e:
        print(f"✗ SbertEmbeddings import failed: {e}")
    
    try:
        # Test semantic chunker
        from bu_processor.semantic.greedy_boundary_chunker import GreedyBoundarySemanticChunker
        print("✓ GreedyBoundarySemanticChunker import works")
    except Exception as e:
        print(f"✗ GreedyBoundarySemanticChunker import failed: {e}")
    
    try:
        # Test structure detection
        from bu_processor.semantic.structure import detect_headings
        print("✓ Structure detection import works")
    except Exception as e:
        print(f"✗ Structure detection import failed: {e}")
    
    try:
        # Test sentence splitting
        from bu_processor.semantic.sentences import sentence_split_with_offsets
        print("✓ Sentence splitting import works")
    except Exception as e:
        print(f"✗ Sentence splitting import failed: {e}")
    
    try:
        # Test PDF extractor
        from bu_processor.pipeline.pdf_extractor import EnhancedPDFExtractor, ChunkingStrategy
        extractor = EnhancedPDFExtractor()
        print("✓ EnhancedPDFExtractor import and instantiation works")
        
        # Check the REAL_SEMANTIC_CHUNKING_AVAILABLE flag
        import bu_processor.pipeline.pdf_extractor as pdf_mod
        print(f"✓ REAL_SEMANTIC_CHUNKING_AVAILABLE: {getattr(pdf_mod, 'REAL_SEMANTIC_CHUNKING_AVAILABLE', 'Not found')}")
        
    except Exception as e:
        print(f"✗ PDF extractor import failed: {e}")
    
    print("\nImport diagnostics completed!")
    return True

if __name__ == "__main__":
    test_imports()
