#!/usr/bin/env python3
"""Test the sentence splitting with page-aware offsets."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from bu_processor.semantic.sentences import (
    sentence_split_with_offsets,
    enhanced_sentence_split_with_offsets,
    group_sentences_by_page,
    find_sentence_boundaries,
    validate_sentence_offsets
)

def test_basic_sentence_splitting():
    """Test basic sentence splitting with page offsets."""
    print("=== Testing Basic Sentence Splitting ===")
    
    # Sample pages with different content
    pages = [
        (1, "This is the first sentence. Here is another sentence! And a third one?"),
        (2, "Page two starts here. It has different content. Multiple sentences exist."),
        (3, "Final page. Short sentences. Done.")
    ]
    
    sentences = sentence_split_with_offsets(pages)
    
    print(f"Split {len(pages)} pages into {len(sentences)} sentences:")
    for i, (sent, page_no, offset) in enumerate(sentences):
        print(f"  {i+1:2d}. Page {page_no}, offset {offset:2d}: '{sent}'")
    
    return sentences

def test_enhanced_splitting():
    """Test enhanced sentence splitting with start/end offsets."""
    print("\n=== Testing Enhanced Sentence Splitting ===")
    
    pages = [
        (1, "First sentence here. Second sentence follows. Third completes the set.")
    ]
    
    enhanced = enhanced_sentence_split_with_offsets(pages)
    
    print("Enhanced splitting with start/end positions:")
    for sent, page_no, start, end in enhanced:
        print(f"  Page {page_no}: [{start:2d}-{end:2d}] '{sent}'")
    
    return enhanced

def test_sentence_grouping():
    """Test grouping sentences by page."""
    print("\n=== Testing Sentence Grouping ===")
    
    pages = [
        (1, "Page one sentence one. Page one sentence two."),
        (2, "Page two sentence one. Page two sentence two. Page two sentence three."),
        (1, "More content for page one. Additional sentence.")  # Mixed pages
    ]
    
    sentences = sentence_split_with_offsets(pages)
    grouped = group_sentences_by_page(sentences)
    
    print("Grouped sentences by page:")
    for page_no in sorted(grouped.keys()):
        print(f"  Page {page_no}:")
        for sent, offset in grouped[page_no]:
            print(f"    Offset {offset:2d}: '{sent}'")

def test_sentence_boundaries():
    """Test finding sentence boundaries."""
    print("\n=== Testing Sentence Boundaries ===")
    
    text = "First sentence. Second sentence! Third sentence? Fourth sentence."
    boundaries = find_sentence_boundaries(text)
    
    print(f"Text: '{text}'")
    print(f"Sentence boundaries at positions: {boundaries}")
    
    # Show what's at each boundary
    for i, pos in enumerate(boundaries):
        context_start = max(0, pos - 10)
        context_end = min(len(text), pos + 10)
        context = text[context_start:context_end]
        marker_pos = pos - context_start
        context_with_marker = context[:marker_pos] + '|' + context[marker_pos:]
        print(f"  Boundary {i+1}: '...{context_with_marker}...'")

def test_offset_validation():
    """Test validation of sentence offsets.""" 
    print("\n=== Testing Offset Validation ===")
    
    page_text = "Valid sentence one. Valid sentence two. Valid sentence three."
    
    # Test valid offsets
    valid_sentences = [
        ("Valid sentence one.", 0),
        ("Valid sentence two.", 20), 
        ("Valid sentence three.", 40)
    ]
    
    is_valid = validate_sentence_offsets(page_text, valid_sentences)
    print(f"Valid offsets test: {'✓ PASS' if is_valid else '✗ FAIL'}")
    
    # Test invalid offsets
    invalid_sentences = [
        ("Wrong sentence.", 0),
        ("Valid sentence two.", 999),  # Out of bounds
    ]
    
    is_invalid = validate_sentence_offsets(page_text, invalid_sentences)
    print(f"Invalid offsets test: {'✓ PASS' if not is_invalid else '✗ FAIL'}")

def test_real_world_document():
    """Test with a realistic document structure."""
    print("\n=== Testing Real-World Document ===")
    
    pages = [
        (1, """1. INTRODUCTION

This document presents our analysis of market trends. The methodology section follows. 
Results are discussed in detail.

1.1 Background

Market research is essential for business strategy. We analyzed multiple data sources. 
The findings are significant."""),
        
        (2, """2. METHODOLOGY  

Our approach involved three main steps. First, we collected data from surveys. 
Second, we analyzed the responses statistically.

2.1 Data Collection

We used online surveys for data collection. Response rates were high. 
The sample size was adequate for analysis."""),
        
        (3, """3. RESULTS

The analysis revealed several key findings. Market growth is accelerating. 
Customer preferences are shifting towards digital solutions.

CONCLUSION

This study provides valuable insights. Further research is recommended. 
Implementation should begin immediately.""")
    ]
    
    sentences = sentence_split_with_offsets(pages)
    grouped = group_sentences_by_page(sentences)
    
    print(f"Processed {len(pages)} pages into {len(sentences)} sentences")
    
    for page_no in sorted(grouped.keys()):
        page_sentences = grouped[page_no]
        print(f"\nPage {page_no}: {len(page_sentences)} sentences")
        
        # Show first few sentences as examples
        for i, (sent, offset) in enumerate(page_sentences[:3]):
            if len(sent) > 60:
                sent_preview = sent[:57] + "..."
            else:
                sent_preview = sent
            print(f"  {i+1}. [{offset:3d}] {sent_preview}")
        
        if len(page_sentences) > 3:
            print(f"  ... and {len(page_sentences) - 3} more sentences")

def test_comparison_with_existing():
    """Compare with existing sentence_split function.""" 
    print("\n=== Comparing with Existing Functionality ===")
    
    try:
        from bu_processor.chunking import sentence_split
        
        text = "First sentence here. Second sentence follows. Third sentence completes."
        
        # Existing function
        existing_sentences = sentence_split(text)
        print(f"Existing sentence_split: {len(existing_sentences)} sentences")
        for i, sent in enumerate(existing_sentences):
            print(f"  {i+1}. '{sent}'")
        
        # Our page-aware version
        pages = [(1, text)]
        our_sentences = sentence_split_with_offsets(pages)
        print(f"\nOur sentence_split_with_offsets: {len(our_sentences)} sentences")
        for i, (sent, page, offset) in enumerate(our_sentences):
            print(f"  {i+1}. Page {page}, offset {offset}: '{sent}'")
        
        # Check consistency
        existing_texts = [s.strip() for s in existing_sentences if s.strip()]
        our_texts = [s.strip() for s, _, _ in our_sentences if s.strip()]
        
        matches = len(existing_texts) == len(our_texts) and all(
            e == o for e, o in zip(existing_texts, our_texts)
        )
        
        print(f"\nConsistency check: {'✓ PASS' if matches else '✗ FAIL'}")
        
    except ImportError:
        print("Could not import existing sentence_split function")

if __name__ == "__main__":
    test_basic_sentence_splitting()
    test_enhanced_splitting()
    test_sentence_grouping()
    test_sentence_boundaries()
    test_offset_validation()
    test_real_world_document()
    test_comparison_with_existing()
    
    print("\n✅ All sentence splitting tests completed!")
