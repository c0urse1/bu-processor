"""
BU-Processor Pipeline Module
===========================

Core document processing pipeline including PDF extraction, 
classification, semantic analysis, and deduplication.
"""

"""
BU-Processor Pipeline Module
===========================

Core document processing pipeline including PDF extraction, 
classification, semantic analysis, and deduplication.
"""

# Core Pipeline Components (always safe to import)
from .classifier import RealMLClassifier
from .pdf_extractor import EnhancedPDFExtractor, ChunkingStrategy
try:
    from .content_types import ContentType
except Exception:  # pragma: no cover
    from enum import Enum as _Enum
    class ContentType(_Enum):  # Minimal fallback
        UNKNOWN = "unknown"

from .enhanced_integrated_pipeline import (
    EnhancedIntegratedPipeline,
    process_documents_multiprocessing,
)

# Chatbot Integration
try:
    from .chatbot_integration import BUProcessorChatbot, ChatbotConfig, ChatbotCLI
    CHATBOT_AVAILABLE = True
except ImportError:
    CHATBOT_AVAILABLE = False

# Security-Enhanced Chatbot
try:
    from .chatbot_security import (
        SecureBUProcessorChatbot,
        SecureChatbotCLI,
        EnhancedChatbotConfig,
    )
    SECURE_CHATBOT_AVAILABLE = True
except ImportError:
    SECURE_CHATBOT_AVAILABLE = False

# Semantic Enhancement
try:
    from .semantic_chunking_enhancement import SemanticClusteringEnhancer
    SEMANTIC_ENHANCEMENT_AVAILABLE = True
except ImportError:
    SEMANTIC_ENHANCEMENT_AVAILABLE = False

# Deduplication (only import on demand to avoid heavy dependencies)
try:
    # Only import when explicitly needed
    DEDUPLICATION_AVAILABLE = True
except ImportError:
    DEDUPLICATION_AVAILABLE = False

def get_semantic_deduplicator():
    """Lazy import of SemanticDeduplicator to avoid heavy dependencies at import time."""
    try:
        from .simhash_semantic_deduplication import SemanticDeduplicator
        return SemanticDeduplicator
    except ImportError:
        return None

# Vector DB Integration  
try:
    from .pinecone_integration import PineconeManager
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    # Dummy class to prevent AttributeError in tests
    class PineconeManager:
        """Dummy PineconeManager when Pinecone is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Pinecone integration not available")
        
        @staticmethod
        def is_available():
            return False

__all__ = [
    # Core Components
    "RealMLClassifier",
    "EnhancedPDFExtractor", 
    "ChunkingStrategy",
    "ContentType",
    "EnhancedIntegratedPipeline",
    "process_documents_multiprocessing",
    # Functions
    "get_semantic_deduplicator",
]

# Add optional components based on availability
if CHATBOT_AVAILABLE:
    __all__.extend([
        "BUProcessorChatbot",
        "ChatbotConfig", 
        "ChatbotCLI",
    ])

if SECURE_CHATBOT_AVAILABLE:
    __all__.extend([
        "SecureBUProcessorChatbot",
        "SecureChatbotCLI",
        "EnhancedChatbotConfig",
    ])

if SEMANTIC_ENHANCEMENT_AVAILABLE:
    __all__.append("SemanticClusteringEnhancer")

# PineconeManager is always available (either real or dummy)
__all__.append("PineconeManager")
