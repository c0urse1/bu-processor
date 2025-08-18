#!/usr/bin/env python3
"""Simple test to isolate import issues."""

print("Testing individual imports...")

try:
    from bu_processor.semantic.embeddings import SbertEmbeddings
    print("✓ SbertEmbeddings import OK")
except Exception as e:
    print(f"✗ SbertEmbeddings import failed: {e}")

try:
    from bu_processor.semantic.testing import FakeDeterministicEmbeddings  
    print("✓ FakeDeterministicEmbeddings import OK")
except Exception as e:
    print(f"✗ FakeDeterministicEmbeddings import failed: {e}")

try:
    from bu_processor.semantic.chunker import semantic_segment_sentences
    print("✓ semantic_segment_sentences import OK")
except Exception as e:
    print(f"✗ semantic_segment_sentences import failed: {e}")

try:
    from bu_processor.semantic.tokens import approx_token_count
    print("✓ approx_token_count import OK")
except Exception as e:
    print(f"✗ approx_token_count import failed: {e}")

try:
    from bu_processor.pipeline.chunk_entry import chunk_document_pages
    print("✓ chunk_document_pages import OK")
except Exception as e:
    print(f"✗ chunk_document_pages import failed: {e}")

print("Import test completed.")
