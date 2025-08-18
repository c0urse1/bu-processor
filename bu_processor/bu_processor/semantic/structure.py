from __future__ import annotations
from typing import List, Tuple
import re

# Heuristics for headings: numbered or ALL CAPS short lines
RE_NUM_HEADING = re.compile(r"^\s*((?:[0-9IVXLC]+(?:\.[0-9IVXLC]+)*)\.?)\s+(.{1,120})$", re.IGNORECASE)
RE_ALL_CAPS = re.compile(r"^[A-Z0-9][A-Z0-9\s\-:,()]{3,80}$")

def detect_headings(page_text: str) -> List[Tuple[int, str, str]]:
    """
    Returns a list of (char_offset, section_number, heading_title) within page_text.
    
    Uses heuristic rules to identify headings:
    1. Numbered headings (e.g., "1.2 Introduction", "III. Overview") 
    2. ALL CAPS lines that are reasonably short (likely headings)
    
    Args:
        page_text: Text content of a single page
        
    Returns:
        List of tuples: (char_offset, section_number, heading_title)
        - char_offset: Position of heading start in page_text
        - section_number: Extracted section number (e.g., "1.2", "III") or empty string
        - heading_title: The heading text content
    """
    out = []
    pos = 0
    for line in page_text.splitlines(keepends=True):
        raw = line.strip()
        m = RE_NUM_HEADING.match(raw)
        if m:
            out.append((pos, m.group(1).rstrip("."), m.group(2).strip()))
        elif RE_ALL_CAPS.match(raw) and len(raw.split()) <= 12:
            out.append((pos, "", raw.title()))
        pos += len(line)
    return out

def assign_section_for_offset(offset: int, page_heads: List[Tuple[int, str, str]]) -> Tuple[str, str]:
    """
    Given a char offset into the page, pick the nearest preceding heading.
    
    This function determines which section a particular piece of text belongs to
    by finding the most recent heading that appears before the given offset.
    
    Args:
        offset: Character position within the page text
        page_heads: List of (char_offset, section_number, heading_title) from detect_headings()
        
    Returns:
        Tuple of (section_number, title). Empty strings if no preceding heading found.
    """
    last = ("", "")
    for off, sec, title in page_heads:
        if off <= offset:
            last = (sec, title)
        else:
            break
    return last

def build_heading_hierarchy(headings: List[Tuple[int, str, str]]) -> List[Tuple[int, str, str, List[str]]]:
    """
    Build a hierarchical structure from detected headings.
    
    Args:
        headings: List of (char_offset, section_number, heading_title) from detect_headings()
        
    Returns:
        List of (char_offset, section_number, heading_title, heading_path) where
        heading_path is a list showing the hierarchical path to this heading
    """
    result = []
    path_stack = []
    
    for offset, section_num, title in headings:
        if not section_num:
            # For non-numbered headings, treat as top-level
            path_stack = [title]
        else:
            # Determine hierarchy level based on section number depth
            depth = len([x for x in section_num.split('.') if x])
            
            # Adjust path stack to current depth
            if depth <= len(path_stack):
                path_stack = path_stack[:depth-1] + [title]
            else:
                # Extend path if needed
                while len(path_stack) < depth - 1:
                    path_stack.append("")
                path_stack.append(title)
        
        result.append((offset, section_num, title, path_stack.copy()))
    
    return result

def extract_section_context(text: str, target_offset: int, max_context: int = 200) -> str:
    """
    Extract surrounding context around a target position for better chunk understanding.
    
    Args:
        text: Full text content
        target_offset: Character position to extract context around
        max_context: Maximum characters to include before and after target
        
    Returns:
        Context string around the target position
    """
    start = max(0, target_offset - max_context)
    end = min(len(text), target_offset + max_context)
    
    # Try to break at word boundaries
    context = text[start:end]
    
    # Clean up partial words at boundaries
    if start > 0 and not text[start].isspace():
        # Find first space to avoid cutting words
        space_pos = context.find(' ')
        if space_pos != -1:
            context = context[space_pos+1:]
    
    if end < len(text) and not text[end-1].isspace():
        # Find last space to avoid cutting words
        space_pos = context.rfind(' ')
        if space_pos != -1:
            context = context[:space_pos]
    
    return context.strip()

def normalize_section_number(section_num: str) -> str:
    """
    Normalize section numbers to a consistent format.
    
    Args:
        section_num: Raw section number (e.g., "1.2.", "III", "2.1.3")
        
    Returns:
        Normalized section number (e.g., "1.2", "3", "2.1.3")
    """
    if not section_num:
        return ""
    
    # Remove trailing dots
    normalized = section_num.rstrip('.')
    
    # Convert Roman numerals to Arabic if needed
    roman_map = {
        'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5',
        'VI': '6', 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10',
        'XI': '11', 'XII': '12', 'XIII': '13', 'XIV': '14', 'XV': '15'
    }
    
    if normalized.upper() in roman_map:
        return roman_map[normalized.upper()]
    
    return normalized
