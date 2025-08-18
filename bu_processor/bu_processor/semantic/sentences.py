from __future__ import annotations
from typing import List, Tuple
import re

# Import the existing sentence splitter to leverage existing logic
try:
    from ..chunking import sentence_split
    CHUNKING_AVAILABLE = True
except ImportError:
    CHUNKING_AVAILABLE = False

# Import NLTK if available (same pattern as PDF extractor)
try:
    import nltk
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    import types
    nltk = types.SimpleNamespace()
    nltk.sent_tokenize = lambda text, language='german': re.split(r'(?<=[.!?])\s+', text)

# Fallback regex pattern
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def sentence_split_with_offsets(pages: List[Tuple[int, str]]) -> List[Tuple[str, int, int]]:
    """
    Split text into sentences with page-aware offsets.
    
    Uses the best available sentence splitter (NLTK > existing chunking > regex fallback)
    while preserving page and offset information for each sentence.
    
    Args:
        pages: List of (page_number, page_text) tuples
        
    Returns:
        List of (sentence, page_number, offset_in_page) tuples
    """
    out = []
    
    for page_no, text in pages:
        if not text.strip():
            continue
            
        # Use the best available sentence splitter
        if NLTK_AVAILABLE:
            sentences = nltk.sent_tokenize(text.strip(), language='german')
        elif CHUNKING_AVAILABLE:
            sentences = sentence_split(text)
        else:
            # Fallback to simple regex
            sentences = SENT_SPLIT.split(text.strip())
        
        # Now find the offset of each sentence in the original page text
        text_cursor = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
                
            # Find this sentence in the page text starting from cursor
            # We need to be careful about whitespace and variations
            sent_start = text.find(sent, text_cursor)
            
            if sent_start == -1:
                # Fallback: try to find a close match or use cursor position
                # This can happen if sentence splitting modified the text slightly
                sent_start = text_cursor
                
                # Try to find the first few words of the sentence
                words = sent.split()[:3]
                if words:
                    partial = ' '.join(words)
                    partial_start = text.find(partial, text_cursor)
                    if partial_start != -1:
                        sent_start = partial_start
            
            out.append((sent, page_no, sent_start))
            text_cursor = sent_start + len(sent)
    
    return out

def enhanced_sentence_split_with_offsets(pages: List[Tuple[int, str]]) -> List[Tuple[str, int, int, int]]:
    """
    Enhanced sentence splitting that also returns end positions.
    
    Args:
        pages: List of (page_number, page_text) tuples
        
    Returns:
        List of (sentence, page_number, start_offset, end_offset) tuples
    """
    basic_results = sentence_split_with_offsets(pages)
    enhanced = []
    
    for sent, page_no, start_offset in basic_results:
        end_offset = start_offset + len(sent)
        enhanced.append((sent, page_no, start_offset, end_offset))
    
    return enhanced

def group_sentences_by_page(sentences_with_offsets: List[Tuple[str, int, int]]) -> dict:
    """
    Group sentences by page number for easier processing.
    
    Args:
        sentences_with_offsets: Output from sentence_split_with_offsets()
        
    Returns:
        Dictionary mapping page_number -> list of (sentence, offset) tuples
    """
    pages = {}
    
    for sent, page_no, offset in sentences_with_offsets:
        if page_no not in pages:
            pages[page_no] = []
        pages[page_no].append((sent, offset))
    
    return pages

def find_sentence_boundaries(text: str, max_boundary_search: int = 50) -> List[int]:
    """
    Find all sentence boundary positions in text.
    
    Useful for mapping sentences back to character positions when chunking.
    
    Args:
        text: Input text to analyze
        max_boundary_search: Maximum characters to look ahead for sentence boundaries
        
    Returns:
        List of character positions where sentences end
    """
    boundaries = []
    
    # Use same sentence splitting logic as above
    if NLTK_AVAILABLE:
        sentences = nltk.sent_tokenize(text.strip(), language='german')
    elif CHUNKING_AVAILABLE:
        sentences = sentence_split(text)
    else:
        sentences = SENT_SPLIT.split(text.strip())
    
    cursor = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
            
        sent_start = text.find(sent, cursor)
        if sent_start != -1:
            sent_end = sent_start + len(sent)
            boundaries.append(sent_end)
            cursor = sent_end
    
    return boundaries

def validate_sentence_offsets(page_text: str, sentences_with_offsets: List[Tuple[str, int]]) -> bool:
    """
    Validate that sentence offsets correctly map back to the original text.
    
    Args:
        page_text: Original page text
        sentences_with_offsets: List of (sentence, offset) for this page
        
    Returns:
        True if all offsets are valid, False otherwise
    """
    for sent, offset in sentences_with_offsets:
        if offset < 0 or offset >= len(page_text):
            return False
            
        # Check if the sentence appears at the expected position
        actual_text = page_text[offset:offset + len(sent)]
        if actual_text.strip() != sent.strip():
            # Allow for minor whitespace differences
            if actual_text.replace(' ', '').replace('\n', '') != sent.replace(' ', '').replace('\n', ''):
                return False
    
    return True
