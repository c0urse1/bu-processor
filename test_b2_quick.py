#!/usr/bin/env python3
"""
Quick diagnostic for B2 metadata flow - simplified version
"""

def test_b2_quick():
    """Quick test to see what metadata is actually being stored."""
    
    print("=== B2 Quick Diagnostic ===")
    
    try:
        from bu_processor.bu_processor.pipeline.pdf_extractor import EnhancedPDFExtractor, ChunkingStrategy
        from bu_processor.bu_processor.factories import make_embedder, make_index, make_store
        from bu_processor.bu_processor.pipeline.upsert_pipeline import convert_chunks_to_dict
        
        # Create sample DocumentChunk-style object to test conversion
        class MockChunk:
            def __init__(self):
                self.text = "Sample text for testing"
                self.chunk_id = "test_chunk_123"
                self.doc_id = "test_doc_456"
                self.page_start = 1
                self.page_end = 1
                self.section = "Test Section"
                self.heading_path = "1.0 Test Section > 1.1 Subsection"
                self.heading_text = "Test Section"
                self.chunk_type = "semantic"
                self.importance_score = 1.5
                self.char_span = (0, 100)
                self.start_position = 0
                self.end_position = 100
                self.meta = {"test_key": "test_value"}
        
        # Test 1: Check convert_chunks_to_dict function
        print("\n--- Test 1: Check chunk conversion ---")
        mock_chunk = MockChunk()
        converted = convert_chunks_to_dict([mock_chunk])
        
        print(f"Converted chunk keys: {list(converted[0].keys())}")
        print(f"- text: {converted[0]['text'][:30]}...")
        print(f"- page: {converted[0]['page']}")
        print(f"- section: {converted[0]['section']}")
        print(f"- meta keys: {list(converted[0]['meta'].keys())}")
        
        # Check if B2 fields are in meta
        meta = converted[0]['meta']
        b2_fields = ['doc_id', 'chunk_id', 'page_start', 'page_end', 'heading_path', 'section']
        print(f"\nB2 fields in meta:")
        for field in b2_fields:
            value = meta.get(field, 'MISSING')
            print(f"  - {field}: {value}")
        
        # Test 2: Check upsert pipeline metadata creation
        print("\n--- Test 2: Check upsert metadata creation ---")
        
        # Simulate what happens in embed_and_index_chunks
        chunk_dict = converted[0]
        
        # This is the metadata creation logic from upsert_pipeline
        md = {
            "doc_id": chunk_dict.get("meta", {}).get("doc_id", "fallback_doc_id"),
            "chunk_id": chunk_dict.get("meta", {}).get("chunk_id", "fallback_chunk_id"),
            "title": "Test Title",
            "source": "test_source.pdf",
            "page": chunk_dict.get("page"),
            "section": chunk_dict.get("section"),
            "page_start": chunk_dict.get("meta", {}).get("page_start"),
            "page_end": chunk_dict.get("meta", {}).get("page_end"),
            "heading_path": chunk_dict.get("meta", {}).get("heading_path"),
            "chunk_type": chunk_dict.get("meta", {}).get("chunk_type"),
            "importance_score": chunk_dict.get("meta", {}).get("importance_score"),
            **(chunk_dict.get("meta") or {}),
        }
        
        print(f"Final metadata for vector storage:")
        for key, value in md.items():
            print(f"  - {key}: {value}")
        
        # Test 3: Check required B2 fields
        print(f"\n--- Test 3: B2 Field Verification ---")
        required_b2 = ["doc_id", "chunk_id", "title", "source", "page_start", "page_end", "section", "heading_path"]
        missing = [field for field in required_b2 if md.get(field) is None]
        present = [field for field in required_b2 if md.get(field) is not None]
        
        print(f"Required B2 fields: {len(required_b2)}")
        print(f"Present: {len(present)} - {present}")
        print(f"Missing: {len(missing)} - {missing}")
        
        if len(missing) == 0:
            print(f"✅ All B2 fields present in metadata!")
            return True
        else:
            print(f"⚠️  Some B2 fields missing")
            return False
        
    except Exception as e:
        print(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_b2_quick()
    if success:
        print(f"\n🎉 B2 Metadata Flow DIAGNOSTIC PASSED!")
    else:
        print(f"\n❌ B2 Metadata Flow DIAGNOSTIC FAILED!")
    
    try:
        from bu_processor.bu_processor.pipeline.upsert_pipeline import convert_chunks_to_dict
        
        # Create a mock DocumentChunk
        class MockChunk:
            def __init__(self):
                self.text = "Sample chunk text for testing B2 metadata flow"
                self.page_start = 2
                self.page_end = 3
                self.section = "2. Analysis"
                self.heading_path = "2 > Analysis > 2.1 Methodology"
                self.heading_text = "Methodology"
                self.chunk_type = "semantic"
                self.importance_score = 1.5
                self.doc_id = "test-doc-b2"
                self.chunk_id = "test-chunk-b2"
                self.start_position = 150
                self.end_position = 280
                self.meta = {"extraction_method": "pymupdf", "tenant": "test_org"}
        
        print("✓ Mock chunk created")
        
        # Test conversion
        mock_chunks = [MockChunk()]
        converted = convert_chunks_to_dict(mock_chunks)
        
        print(f"✓ Chunks converted: {len(converted)}")
        
        chunk_dict = converted[0]
        metadata = chunk_dict['meta']
        
        print(f"\nB2 Metadata Structure Check:")
        print(f"  - text: {chunk_dict['text'][:50]}...")
        print(f"  - page (legacy): {chunk_dict['page']}")
        print(f"  - section: {chunk_dict['section']}")
        
        print(f"\nB2 Enhanced Metadata in 'meta':")
        b2_fields = {
            'page_start': metadata.get('page_start'),
            'page_end': metadata.get('page_end'), 
            'heading_path': metadata.get('heading_path'),
            'chunk_type': metadata.get('chunk_type'),
            'importance_score': metadata.get('importance_score'),
            'heading_text': metadata.get('heading_text'),
            'start_position': metadata.get('start_position'),
            'end_position': metadata.get('end_position')
        }
        
        for field, value in b2_fields.items():
            status = "✅" if value is not None else "❌"
            print(f"  {status} {field}: {value}")
        
        # Count present fields
        present_count = sum(1 for v in b2_fields.values() if v is not None)
        total_count = len(b2_fields)
        
        print(f"\nB2 Summary:")
        print(f"  - Present fields: {present_count}/{total_count}")
        print(f"  - Success rate: {present_count/total_count*100:.1f}%")
        
        if present_count >= 6:  # Most fields should be present
            print(f"  - Status: ✅ B2 METADATA STRUCTURE WORKING")
            return True
        else:
            print(f"  - Status: ❌ B2 METADATA INCOMPLETE")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_b2_metadata_structure()
    if success:
        print(f"\n🎉 B2 metadata structure test PASSED!")
    else:
        print(f"\n❌ B2 metadata structure test FAILED!")
