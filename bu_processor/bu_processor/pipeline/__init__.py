"""
BU-Processor Pipeline Module
===========================

Core document processing pipeline including PDF extraction, 
classification, semantic analysis, and deduplication.

This module uses lazy imports to avoid heavy dependencies at import time.
Import specific components as needed.
"""

# Define available components without eager imports
__all__ = [
    "enhanced_integrated_pipeline",
    "pdf_extractor", 
    "classifier",
    "content_types",
    "pinecone_integration",
    "chatbot_integration",
    "semantic_chunking_enhancement",
    "simhash_semantic_deduplication",
]

# Lazy import helpers for backwards compatibility
def get_classifier():
    """Lazy import of RealMLClassifier."""
    from .classifier import RealMLClassifier
    return RealMLClassifier

def get_pdf_extractor():
    """Lazy import of EnhancedPDFExtractor."""
    from .pdf_extractor import EnhancedPDFExtractor
    return EnhancedPDFExtractor

def get_semantic_deduplicator():
    """Lazy import of SemanticDeduplicator to avoid heavy dependencies at import time."""
    try:
        from .simhash_semantic_deduplication import SemanticDeduplicator
        return SemanticDeduplicator
    except ImportError:
        return None

def get_pinecone_manager():
    """Lazy import of PineconeManager."""
    try:
        from .pinecone_integration import PineconeManager
        return PineconeManager
    except ImportError:
        return None

def get_chatbot_integration():
    """Lazy import of ChatbotIntegration."""
    try:
        from .chatbot_integration import ChatbotIntegration
        return ChatbotIntegration
    except ImportError:
        return None

# Backwards compatibility for tests that expect immediate imports
# Only import core components that are always needed
try:
    from .classifier import RealMLClassifier
except ImportError:
    pass

try:
    from .pdf_extractor import EnhancedPDFExtractor
except ImportError:
    pass
