# bu_processor/integrations/pinecone_manager.py
"""
Pinecone integration main entry point.

This module provides the primary PineconeManager interface that automatically
selects between simple and enhanced implementations based on configuration.

For direct access to specific implementations:
- pinecone_simple.py: Simple, stable MVP implementation
- pinecone_enhanced.py: Advanced features (when implemented)
- pinecone_facade.py: Facade pattern with automatic selection

This file re-exports the facade for backward compatibility.
"""

# Import the facade implementation
from .pinecone_facade import PineconeManager, get_pinecone_manager

# For explicit access to implementations
from .pinecone_simple import PineconeManager as SimplePineconeManager
try:
    from .pinecone_enhanced import PineconeEnhancedManager
except ImportError:
    # Enhanced implementation may not be available
    PineconeEnhancedManager = None

# Export the facade as the default PineconeManager
__all__ = [
    "PineconeManager",           # Facade (default)
    "get_pinecone_manager",      # Factory function
    "SimplePineconeManager",     # Direct access to simple impl
    "PineconeEnhancedManager"    # Direct access to enhanced impl (if available)
]
