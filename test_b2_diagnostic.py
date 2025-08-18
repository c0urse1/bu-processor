#!/usr/bin/env python3
"""
Quick diagnostic for B2 metadata flow verification.
"""

def test_b2_metadata_flow():
    """Test that B2 enhanced metadata fields flow correctly."""
    
    print("=== B2 Metadata Flow Diagnostic ===")
    
    try:
        from bu_processor.bu_processor.pipeline.upsert_pipeline import convert_chunks_to_dict
        
        # Create a mock DocumentChunk to test conversion
        class MockChunk:
            def __init__(self):
                self.text = "Sample text for B2 testing"
                self.chunk_id = "test_chunk_123"
                self.doc_id = "test_doc_456"
                self.page_start = 1
                self.page_end = 2
                self.section = "1.0 Introduction"
                self.heading_path = "1.0 Introduction > 1.1 Background"
                self.heading_text = "Background"
                self.chunk_type = "semantic"
                self.importance_score = 1.5
                self.meta = {"additional": "metadata"}
        
        print("\n--- Step 1: Test convert_chunks_to_dict ---")
        mock_chunk = MockChunk()
        converted = convert_chunks_to_dict([mock_chunk])
        chunk_dict = converted[0]
        
        print(f"Converted keys: {list(chunk_dict.keys())}")
        print(f"- text: present ({len(chunk_dict['text'])} chars)")
        print(f"- page: {chunk_dict['page']}")
        print(f"- section: {chunk_dict['section']}")
        
        # Check meta structure
        meta = chunk_dict.get('meta', {})
        print(f"- meta keys: {list(meta.keys())}")
        
        print("\n--- Step 2: Check B2 fields in meta ---")
        b2_meta_fields = ['doc_id', 'chunk_id', 'page_start', 'page_end', 'heading_path']
        for field in b2_meta_fields:
            value = meta.get(field, 'MISSING')
            status = "✓" if value != 'MISSING' else "❌"
            print(f"  {status} {field}: {value}")
        
        print("\n--- Step 3: Simulate upsert metadata creation ---")
        # This simulates the metadata dict creation in embed_and_index_chunks
        final_metadata = {
            "doc_id": meta.get("doc_id", "fallback"),
            "chunk_id": meta.get("chunk_id", "fallback"), 
            "title": "Test Document",
            "source": "test.pdf",
            "page": chunk_dict.get("page"),
            "section": chunk_dict.get("section"),
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "heading_path": meta.get("heading_path"),
            "chunk_type": meta.get("chunk_type"),
            "importance_score": meta.get("importance_score"),
        }
        
        # Add any additional meta fields
        final_metadata.update(meta)
        
        print("Final vector metadata:")
        for key, value in final_metadata.items():
            print(f"  - {key}: {value}")
        
        print("\n--- Step 4: B2 Compliance Check ---")
        required_b2_fields = ["doc_id", "chunk_id", "title", "source", "page_start", "page_end", "section", "heading_path"]
        missing = [f for f in required_b2_fields if final_metadata.get(f) is None]
        present = [f for f in required_b2_fields if final_metadata.get(f) is not None]
        
        print(f"B2 Required fields: {len(required_b2_fields)}")
        print(f"Present: {len(present)} - {present}")
        print(f"Missing: {len(missing)} - {missing}")
        
        if len(missing) == 0:
            print("✅ ALL B2 FIELDS PRESENT - metadata flow working correctly!")
            return True
        else:
            print(f"⚠️  MISSING B2 FIELDS: {missing}")
            return False
        
    except Exception as e:
        print(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_b2_metadata_flow()
    print(f"\n{'🎉 B2 DIAGNOSTIC PASSED!' if success else '❌ B2 DIAGNOSTIC FAILED!'}")
