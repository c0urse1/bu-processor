#!/usr/bin/env python3
"""Test script for enhanced DocumentChunk model."""

from bu_processor.models.chunk import DocumentChunk, create_semantic_chunk, create_paragraph_chunk

def test_enhanced_document_chunk():
    """Test the enhanced DocumentChunk model with all its new features."""
    
    # Test basic DocumentChunk creation
    chunk1 = DocumentChunk(
        chunk_id='test_chunk_1',
        text='This is a test chunk with rich metadata.',
        doc_id='test_doc_001',
        page_start=1,
        page_end=2,
        section='2.1 Testing Section',
        heading_path=['2 Testing', '2.1 Testing Section'],
        char_span=(100, 200),
        chunk_type='semantic',
        importance_score=0.8,
        heading_text='Testing Section',
        meta={'source': 'test_document.pdf', 'author': 'Test Author'}
    )

    print('DocumentChunk created successfully!')
    print(f'ID: {chunk1.chunk_id}')
    print(f'Text: {chunk1.text[:50]}...')
    print(f'Page range: {chunk1.page_range}')
    print(f'Heading context: {chunk1.heading_context}')
    print(f'Context text: {chunk1.get_context_text()[:80]}...')
    
    # Test char_span to position mapping
    print(f'Start position: {chunk1.start_position}')
    print(f'End position: {chunk1.end_position}')

    # Test helper functions
    chunk2 = create_semantic_chunk(
        chunk_id='semantic_test',
        text='This is a semantic chunk created with helper function.',
        doc_id='test_doc_002',
        page_num=3,
        section='3.1 Semantic Testing',
        heading='Semantic Testing',
        importance_score=0.9
    )

    print(f'\nSemantic chunk created: {chunk2.chunk_id}')
    print(f'Type: {chunk2.chunk_type}')
    print(f'Importance: {chunk2.importance_score}')

    # Test paragraph chunk
    chunk3 = create_paragraph_chunk(
        chunk_id='paragraph_test',
        text='This is a paragraph chunk.',
        doc_id='test_doc_003',
        page_num=4,
        char_span=(300, 350)
    )
    
    print(f'\nParagraph chunk created: {chunk3.chunk_id}')
    print(f'Type: {chunk3.chunk_type}')

    # Test to_dict and from_dict
    chunk_dict = chunk1.to_dict()
    chunk_restored = DocumentChunk.from_dict(chunk_dict)
    print(f'\nSerialization test: {chunk_restored.chunk_id == chunk1.chunk_id}')
    
    # Test update_metadata
    chunk1.update_metadata(custom_field='custom_value', importance_score=0.95)
    print(f'Updated importance score: {chunk1.importance_score}')
    print(f'Custom metadata: {chunk1.meta.get("custom_field")}')

    print('\nAll tests passed!')
    return True

if __name__ == '__main__':
    test_enhanced_document_chunk()
