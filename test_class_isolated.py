#!/usr/bin/env python3
"""
Simple test to check just the class definition without imports
"""

import sys
sys.path.insert(0, r"c:\ml_classifier_poc\bu_processor")

print("Testing isolated class creation...")

# Read the file and extract just the class definition
file_path = r"c:\ml_classifier_poc\bu_processor\bu_processor\pipeline\semantic_chunking_enhancement.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the class definition and extract it
lines = content.split('\n')
class_start = None
class_code = []

for i, line in enumerate(lines):
    if 'SemanticClusteringEnhancer' in line:
        print(f"Line {i+1}: '{line.strip()}'")
    if line.strip().startswith('class SemanticClusteringEnhancer'):
        class_start = i
        class_indent = len(line) - len(line.lstrip())
        print(f"Found class at line {i+1}, indent: {class_indent}")
        break

if class_start is not None:
    # Extract the class definition
    collecting = True
    for i in range(class_start, len(lines)):
        line = lines[i]
        
        # If we hit another class or function at the same level, stop
        if i > class_start and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            if line.startswith('class ') or line.startswith('def ') or line.startswith('@'):
                break
        
        class_code.append(line)
    
    print(f"Extracted {len(class_code)} lines of class code")
    
    # Create minimal dependencies
    dependencies = """
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Mock dependencies
class ContentType(Enum):
    LEGAL_TEXT = "legal_text"
    TECHNICAL = "technical"

class HierarchicalChunk:
    def __init__(self, id: str = "", text: str = "", metadata: Optional[Dict[str, Any]] = None, importance_score: float = 1.0):
        self.id = id
        self.text = text
        self.metadata = metadata or {}
        self.importance_score = importance_score

def get_logger(name):
    return logging.getLogger(name)

def log_context(**kwargs):
    def decorator(func):
        return func
    return decorator

def timed_operation(description):
    def decorator(func):
        return func
    return decorator

# Feature flags
_HAS_SBERT = False
_HAS_SKLEARN = False
SEMANTIC_ENHANCEMENT_AVAILABLE = False

# Mock missing classes
class ChunkProtocol:
    pass

@dataclass
class ClusterResult:
    cluster_assignments: List[int]
    cluster_centers: Optional[List[List[float]]] = None
    silhouette_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass 
class SemanticSimilarity:
    score: float
    method: str
    metadata: Optional[Dict[str, Any]] = None

class ClusteringMethod(Enum):
    SEMANTIC_KMEANS = "semantic_kmeans"
    SEMANTIC_DBSCAN = "semantic_dbscan"
    TFIDF_CLUSTERING = "tfidf_clustering"
    FALLBACK_SIMPLE = "fallback_simple"
"""
    
    # Try to execute
    try:
        exec(dependencies)
        class_code_str = '\n'.join(class_code)
        exec(class_code_str)
        
        print("✅ Class definition executed successfully!")
        print("Testing instantiation...")
        
        enhancer = SemanticClusteringEnhancer()  # type: ignore
        print("✅ Class instantiated successfully!")
        print(f"Available methods: {[m for m in dir(enhancer) if not m.startswith('_')]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ Class definition not found")
