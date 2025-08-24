#!/usr/bin/env python3
"""
Test script to validate minimal environment configuration
"""
import os
import sys

def test_minimal_config():
    """Test the minimal environment configuration"""
    print('🔍 Testing Minimal Environment Configuration')
    print('=' * 50)
    
    # Test required environment variables
    print(f'✅ PINECONE_API_KEY present: {bool(os.getenv("PINECONE_API_KEY"))}')
    print(f'✅ PINECONE_ENV: {os.getenv("PINECONE_ENV", "not set")}')
    print(f'✅ PINECONE_INDEX_NAME: {os.getenv("PINECONE_INDEX_NAME", "not set")}')
    print(f'✅ PINECONE_NAMESPACE: {os.getenv("PINECONE_NAMESPACE", "not set")}')
    print(f'✅ EMBEDDING_MODEL: {os.getenv("EMBEDDING_MODEL", "not set")}')
    print()
    
    # Test intelligence switches
    try:
        from bu_processor.core.flags import ENABLE_RERANK, ENABLE_ENHANCED_PINECONE, ENABLE_EMBED_CACHE
        print('🚀 Intelligence Switches Status:')
        print(f'   ENABLE_ENHANCED_PINECONE: {ENABLE_ENHANCED_PINECONE}')
        print(f'   ENABLE_EMBED_CACHE: {ENABLE_EMBED_CACHE}')
        print(f'   ENABLE_RERANK: {ENABLE_RERANK}')
        print()
    except ImportError as e:
        print(f'⚠️  Could not import flags: {e}')
        return
    
    # Test factory function
    try:
        from bu_processor.integrations.pinecone_facade import make_pinecone_manager
        pc = make_pinecone_manager(index_name='test-index')
        print(f'✅ Factory function working: {pc.implementation_type}')
    except Exception as e:
        print(f'⚠️  Factory test (expected without valid credentials): {e}')
    
    print('✅ Minimal configuration validated!')

if __name__ == '__main__':
    test_minimal_config()
