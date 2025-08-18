#!/usr/bin/env python3
"""Test the heading and page detection helpers."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from bu_processor.semantic.structure import (
    detect_headings, 
    assign_section_for_offset,
    build_heading_hierarchy,
    extract_section_context,
    normalize_section_number
)

def test_heading_detection():
    """Test basic heading detection functionality."""
    print("=== Testing Heading Detection ===")
    
    # Sample page text with various heading formats
    page_text = """This is some introductory text.

1. INTRODUCTION
This section introduces the topic.

1.1 Background Information  
Some background details here.

2. METHODOLOGY
This explains our approach.

2.1 Data Collection
Details about data collection.

2.2 Analysis Methods
How we analyze the data.

III. RESULTS AND DISCUSSION
The results are presented here.

CONCLUSION
Final thoughts and summary.
"""
    
    headings = detect_headings(page_text)
    print(f"Detected {len(headings)} headings:")
    for offset, section_num, title in headings:
        print(f"  Offset {offset:3d}: Section '{section_num}' - '{title}'")
    
    # Test section assignment
    print("\n=== Testing Section Assignment ===")
    test_offsets = [50, 150, 250, 350, 450]
    for offset in test_offsets:
        section, title = assign_section_for_offset(offset, headings)
        char_at_offset = page_text[offset] if offset < len(page_text) else '(end)'
        print(f"  Offset {offset} ('{char_at_offset}'): Section '{section}' - '{title}'")

def test_heading_hierarchy():
    """Test hierarchical heading structure building."""
    print("\n=== Testing Heading Hierarchy ===")
    
    headings = [
        (0, "1", "Introduction"),
        (100, "1.1", "Background"),
        (200, "1.2", "Objectives"), 
        (300, "2", "Methods"),
        (400, "2.1", "Data Collection"),
        (500, "2.1.1", "Survey Design"),
        (600, "2.2", "Analysis"),
        (700, "3", "Results"),
        (800, "", "Summary")  # Non-numbered heading
    ]
    
    hierarchy = build_heading_hierarchy(headings)
    print("Hierarchical structure:")
    for offset, section_num, title, path in hierarchy:
        indent = "  " * (len(path) - 1)
        print(f"  {indent}{section_num} {title} -> Path: {' > '.join(path)}")

def test_section_normalization():
    """Test section number normalization.""" 
    print("\n=== Testing Section Normalization ===")
    
    test_numbers = ["1.", "2.1.", "III", "IV.", "1.2.3", "V.1", ""]
    for num in test_numbers:
        normalized = normalize_section_number(num)
        print(f"  '{num}' -> '{normalized}'")

def test_context_extraction():
    """Test context extraction around target positions."""
    print("\n=== Testing Context Extraction ===")
    
    text = "This is a long document with multiple sentences. It contains various topics and sections. We want to extract meaningful context around specific positions to better understand the surrounding content and improve chunk quality."
    
    test_positions = [20, 60, 120]
    for pos in test_positions:
        context = extract_section_context(text, pos, max_context=30)
        print(f"  Position {pos}: '{context}'")

def test_real_world_example():
    """Test with a more realistic document excerpt."""
    print("\n=== Testing Real-World Example ===")
    
    document = """
1. EXECUTIVE SUMMARY

This report presents findings from our comprehensive analysis of market trends
and competitive positioning in the technology sector.

1.1 Key Findings

Our analysis reveals three major trends:
- Increased adoption of cloud technologies
- Growing emphasis on data security
- Shift towards remote work solutions

1.2 Recommendations

Based on our findings, we recommend the following strategic initiatives:

2. MARKET ANALYSIS

The technology market has experienced significant changes over the past year.

2.1 Market Size and Growth

Current market size is estimated at $2.5 trillion, with projected growth of 8.5%
annually over the next five years.

2.2 Competitive Landscape

Major players include established companies and emerging startups competing
for market share.

APPENDIX A: METHODOLOGY

This appendix describes our research methodology and data collection procedures.
"""
    
    headings = detect_headings(document)
    print(f"Detected {len(headings)} headings in real-world example:")
    
    hierarchy = build_heading_hierarchy(headings)
    for offset, section_num, title, path in hierarchy:
        section_info, _ = assign_section_for_offset(offset + 50, headings)  # Test content within section
        print(f"  {section_num} {title}")
        print(f"    Path: {' > '.join(path)}")
        
        # Extract some content from this section
        next_heading_offset = len(document)
        for next_offset, _, _ in headings:
            if next_offset > offset:
                next_heading_offset = next_offset
                break
        
        section_content = document[offset:min(offset + 200, next_heading_offset)]
        first_line = section_content.split('\n')[0].strip()
        if len(first_line) > 80:
            first_line = first_line[:77] + "..."
        print(f"    Content preview: {first_line}")
        print()

if __name__ == "__main__":
    test_heading_detection()
    test_heading_hierarchy()
    test_section_normalization()
    test_context_extraction()
    test_real_world_example()
    
    print("\n✅ All heading detection tests completed successfully!")
