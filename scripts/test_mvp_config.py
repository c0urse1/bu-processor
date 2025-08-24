#!/usr/bin/env python3
"""
🔧 MVP CONFIGURATION TEST
========================
Tests the clean MVP configuration setup with minimal variables.
"""

import os
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_mvp_config():
    """Test MVP configuration loading."""
    print("🔧 MVP CONFIGURATION TEST")
    print("=" * 50)
    
    # Test environment variables loading
    print("\n📋 Environment Variables:")
    env_vars = [
        "PINECONE_API_KEY",
        "PINECONE_ENV", 
        "PINECONE_INDEX_NAME",
        "PINECONE_NAMESPACE",
        "EMBEDDING_MODEL",
        "VECTOR_DB_ENABLE",
        "ENABLE_METRICS",
        "ENABLE_EMBEDDING_CACHE",
        "ENABLE_THREADPOOL"
    ]
    
    for var in env_vars:
        value = os.getenv(var, "Not set")
        if "KEY" in var and value != "Not set":
            value = f"{value[:8]}..." if len(value) > 8 else value
        print(f"  {var:<25}: {value}")
    
    # Test config loading
    try:
        from bu_processor.core.config import VectorDatabaseConfig
        print("\n⚙️ Loading VectorDatabaseConfig...")
        
        config = VectorDatabaseConfig()
        
        print(f"  pinecone_api_key       : {'SET' if config.pinecone_api_key else 'NOT SET'}")
        print(f"  pinecone_env           : {config.pinecone_env}")
        print(f"  pinecone_index_name    : {config.pinecone_index_name}")
        print(f"  pinecone_namespace     : {config.pinecone_namespace}")
        print(f"  embedding_model        : {config.embedding_model}")
        print(f"  embedding_dim          : {config.embedding_dim}")
        print(f"  enable_vector_db       : {config.enable_vector_db}")
        print(f"  enable_metrics         : {config.enable_metrics}")
        print(f"  enable_embedding_cache : {config.enable_embedding_cache}")
        print(f"  enable_threadpool      : {config.enable_threadpool}")
        
        print("\n✅ Configuration loaded successfully!")
        
    except Exception as e:
        print(f"\n❌ Configuration loading failed: {e}")
        return False
    
    # Test MVP feature flags
    try:
        from bu_processor.core.mvp_features import MVPFeatureFlags
        print("\n🎯 MVP Feature Flags:")
        
        flags = MVPFeatureFlags.get_all_flags()
        for flag, value in flags.items():
            print(f"  {flag:<25}: {value}")
            
        print("\n✅ Feature flags loaded successfully!")
        
    except Exception as e:
        print(f"\n❌ Feature flags loading failed: {e}")
        return False
    
    return True

def show_sample_env():
    """Show sample .env configuration."""
    print("\n📄 SAMPLE .env CONFIGURATION:")
    print("=" * 50)
    
    sample_env = """# Essential MVP variables
PINECONE_API_KEY=pcsk_your_api_key_here
PINECONE_ENV=us-west1-gcp
PINECONE_INDEX_NAME=bu-processor
PINECONE_NAMESPACE=bu
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
VECTOR_DB_ENABLE=true

# MVP Feature flags (all disabled for simplicity)
ENABLE_METRICS=false
ENABLE_EMBEDDING_CACHE=false
ENABLE_THREADPOOL=false"""
    
    print(sample_env)

if __name__ == "__main__":
    success = test_mvp_config()
    
    if not success:
        show_sample_env()
        print("\n💡 Create a .env file with the above configuration to get started.")
    else:
        print("\n🎉 MVP configuration is working correctly!")
