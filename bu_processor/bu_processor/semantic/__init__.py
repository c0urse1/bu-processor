# bu_processor/semantic/__init__.py
"""Semantic processing components for BU-Processor."""

from .embeddings import EmbeddingsBackend, SbertEmbeddings
from .testing import FakeDeterministicEmbeddings
from .chunker import semantic_segment_sentences
from .tokens import approx_token_count
from .structure import (
    detect_headings, 
    assign_section_for_offset,
    build_heading_hierarchy,
    extract_section_context,
    normalize_section_number
)
from .sentences import (
    sentence_split_with_offsets,
    enhanced_sentence_split_with_offsets,
    group_sentences_by_page,
    find_sentence_boundaries,
    validate_sentence_offsets
)
from .greedy_boundary_chunker import GreedyBoundarySemanticChunker

__all__ = [
    "EmbeddingsBackend",
    "SbertEmbeddings", 
    "FakeDeterministicEmbeddings",
    "semantic_segment_sentences",
    "approx_token_count",
    "detect_headings",
    "assign_section_for_offset", 
    "build_heading_hierarchy",
    "extract_section_context",
    "normalize_section_number",
    "sentence_split_with_offsets",
    "enhanced_sentence_split_with_offsets", 
    "group_sentences_by_page",
    "find_sentence_boundaries",
    "validate_sentence_offsets",
    "GreedyBoundarySemanticChunker"
]
