"""
BU-Processor - Advanced ML Document Classification System
========================================================

A comprehensive ML-powered document classifier and processor specifically 
designed for insurance documents (Berufsunfähigkeitsversicherung).

Features:
- PDF text extraction with multiple fallback methods
- ML-based document classification
- Semantic chunking and deduplication
- Vector database integration (Pinecone)
- REST API with FastAPI
- Interactive chatbot interface
- Comprehensive evaluation metrics

Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "BU-Processor Team"
__email__ = "team@bu-processor.local"

from .core.config import get_config  # lazy loading function only
from .pipeline.simhash_semantic_deduplication import (
    EnhancedDeduplicationEngine,
    SemanticDuplicateDetector,
    SemanticSimHashGenerator,
)
# Convenient imports - lazy loaded
def get_settings():
    """Get settings instance lazily."""
    return get_config()

config = get_settings  # function reference for lazy loading

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "get_settings",
    "config",
    "get_config",
    "EnhancedDeduplicationEngine",
    "SemanticDuplicateDetector",
    "SemanticSimHashGenerator",
]
