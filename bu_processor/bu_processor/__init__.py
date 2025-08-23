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
- Centralized structured logging with structlog

Version: 0.1.0
"""

__version__ = "0.1.0"

# ============================================================================
# CENTRALIZED LOGGING INITIALIZATION
# ============================================================================
# Initialize structured logging as the first action when package is imported.
# This ensures all components use the same logging configuration.

try:
    from .core.logging_setup import configure_logging, get_logger, get_logging_config
    
    # Configure logging immediately on package import
    configure_logging()
    
    # Get a logger to confirm setup
    _init_logger = get_logger(__name__)
    _config_info = get_logging_config()
    
    _init_logger.info("BU-Processor package initialized", 
                     version=__version__,
                     logging_config=_config_info)
    
except ImportError as e:
    # Fallback if logging setup fails
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _fallback_logger = logging.getLogger(__name__)
    _fallback_logger.warning(f"Could not configure structured logging: {e}")
    _fallback_logger.info(f"BU-Processor {__version__} initialized with basic logging")
except Exception as e:
    # Catch any other configuration errors
    import logging
    logging.basicConfig(level=logging.INFO)
    _error_logger = logging.getLogger(__name__)
    _error_logger.error(f"Logging configuration failed: {e}")
    _error_logger.info(f"BU-Processor {__version__} initialized with minimal logging")
__author__ = "BU-Processor Team"
__email__ = "team@bu-processor.local"

from .core.config import get_config  # lazy loading function only
from .pipeline.simhash_semantic_deduplication import (
    EnhancedDeduplicationEngine,
    SemanticDuplicateDetector,
    SemanticSimHashGenerator,
)
from .pipeline.enhanced_integrated_pipeline import EnhancedIntegratedPipeline
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
    "EnhancedIntegratedPipeline",
]
