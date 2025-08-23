#!/usr/bin/env python3
"""
🌲 PINECONE INDEX VALIDATION SCRIPT
==================================

Validates Pinecone index status and performs sample queries to verify
that the vector database is working correctly with your BU-Processor system.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # Import Pinecone
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    print("❌ Pinecone library not installed. Install with: pip install pinecone-client")
    PINECONE_AVAILABLE = False

try:
    # Import BU-Processor components
    from bu_processor.core.config import get_config
    from bu_processor.factories import make_embedder
    BU_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  BU-Processor modules not available: {e}")
    BU_PROCESSOR_AVAILABLE = False

# Fallback embedding model
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers not available for fallback embedding")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configuration
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "bu-processor-embeddings")
SAMPLE_QUERIES = [
    "abstrakte Verweisung in der BU",
    "Bedingungen und Vereinbarungen",
    "Versicherungsschutz bei Schadensfällen",
    "Ausschlüsse und Begrenzungen",
    "Kündigungsbestimmungen"
]

def get_pinecone_client():
    """Initialize Pinecone client"""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    return Pinecone(api_key=api_key)

def check_index_stats(pc: Pinecone, index_name: str) -> Dict[str, Any]:
    """Get and display Pinecone index statistics"""
    print(f"🔍 Checking index: {index_name}")
    
    try:
        # Get index
        index = pc.Index(index_name)
        
        # Get statistics
        stats = index.describe_index_stats()
        
        print(f"📊 Index Statistics:")
        print(f"   Total vectors: {stats.total_vector_count:,}")
        print(f"   Dimension: {stats.dimension}")
        
        # Check if there are any namespaces
        if hasattr(stats, 'namespaces') and stats.namespaces:
            print(f"   Namespaces: {len(stats.namespaces)}")
            for namespace, ns_stats in stats.namespaces.items():
                ns_name = namespace if namespace else "default"
                print(f"     - {ns_name}: {ns_stats.vector_count:,} vectors")
        else:
            print(f"   Namespaces: Using default namespace")
        
        return stats
        
    except Exception as e:
        print(f"❌ Error getting index stats: {e}")
        raise

def get_embedder():
    """Get embedder (BU-Processor factory or fallback)"""
    if BU_PROCESSOR_AVAILABLE:
        try:
            print("🧠 Using BU-Processor embedder...")
            return make_embedder(), "bu_processor"
        except Exception as e:
            print(f"⚠️  BU-Processor embedder failed: {e}")
    
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("🧠 Using fallback sentence-transformers embedder...")
        # Use multilingual model that works well with German business documents
        model = SentenceTransformer("distiluse-base-multilingual-cased")
        return model, "sentence_transformers"
    
    raise ValueError("No embedder available - install sentence-transformers or fix BU-Processor config")

def embed_query(query: str, embedder, embedder_type: str) -> List[float]:
    """Generate embedding for query"""
    try:
        if embedder_type == "bu_processor":
            # Use BU-Processor embedder
            if hasattr(embedder, 'embed_text'):
                return embedder.embed_text(query)
            elif hasattr(embedder, 'encode'):
                return embedder.encode([query])[0].tolist()
            else:
                raise ValueError("Unknown BU-Processor embedder interface")
        
        elif embedder_type == "sentence_transformers":
            # Use sentence-transformers
            return embedder.encode([query])[0].tolist()
        
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")
            
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        raise

def perform_sample_queries(pc: Pinecone, index_name: str, queries: List[str]):
    """Perform sample queries on the index"""
    print(f"\n🔍 Performing sample queries...")
    
    try:
        # Get embedder
        embedder, embedder_type = get_embedder()
        print(f"   Using embedder: {embedder_type}")
        
        # Get index
        index = pc.Index(index_name)
        
        for i, query in enumerate(queries, 1):
            print(f"\n📝 Query {i}: '{query}'")
            
            try:
                # Generate embedding
                query_vector = embed_query(query, embedder, embedder_type)
                print(f"   ✅ Generated embedding (dim: {len(query_vector)})")
                
                # Perform query
                response = index.query(
                    vector=query_vector,
                    top_k=3,
                    include_metadata=True
                )
                
                matches = response.get('matches', [])
                print(f"   📊 Found {len(matches)} matches:")
                
                if not matches:
                    print("     (No matches found)")
                else:
                    for j, match in enumerate(matches, 1):
                        score = match.get('score', 0.0)
                        match_id = match.get('id', 'unknown')
                        metadata = match.get('metadata', {})
                        
                        # Extract relevant metadata
                        filename = metadata.get('file_name', metadata.get('filename', 'unknown'))
                        chunk_info = metadata.get('chunk_id', metadata.get('chunk_index', ''))
                        text_preview = metadata.get('chunk_text', metadata.get('text', ''))
                        
                        print(f"     {j}. Score: {score:.3f}")
                        print(f"        ID: {match_id}")
                        print(f"        File: {filename}")
                        if chunk_info:
                            print(f"        Chunk: {chunk_info}")
                        if text_preview:
                            preview = text_preview[:100] + "..." if len(text_preview) > 100 else text_preview
                            print(f"        Preview: {preview}")
                
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
                continue
    
    except Exception as e:
        print(f"❌ Sample queries failed: {e}")
        raise

def check_index_health(pc: Pinecone, index_name: str) -> bool:
    """Perform basic health checks on the index"""
    print(f"\n🩺 Health check for index: {index_name}")
    
    try:
        # Check if index exists
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]
        
        if index_name not in index_names:
            print(f"❌ Index '{index_name}' not found!")
            print(f"   Available indexes: {index_names}")
            return False
        
        print(f"✅ Index exists")
        
        # Get index details
        index_info = pc.describe_index(index_name)
        print(f"✅ Index details:")
        print(f"   Status: {index_info.status.ready}")
        print(f"   Dimension: {index_info.dimension}")
        print(f"   Metric: {index_info.metric}")
        
        if not index_info.status.ready:
            print(f"⚠️  Index is not ready yet")
            return False
        
        # Get basic stats
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        if stats.total_vector_count == 0:
            print(f"⚠️  Index is empty (no vectors)")
            return False
        
        print(f"✅ Index contains {stats.total_vector_count:,} vectors")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def main():
    """Main validation function"""
    print("🌲 Pinecone Index Validation Script")
    print("=" * 50)
    
    if not PINECONE_AVAILABLE:
        print("❌ Pinecone library not available")
        sys.exit(1)
    
    try:
        # Initialize Pinecone
        print("🔧 Initializing Pinecone client...")
        pc = get_pinecone_client()
        print("✅ Pinecone client initialized")
        
        # Health check
        is_healthy = check_index_health(pc, INDEX_NAME)
        if not is_healthy:
            print("\n❌ Index health check failed")
            sys.exit(1)
        
        # Get detailed stats
        stats = check_index_stats(pc, INDEX_NAME)
        
        # Perform sample queries if index has data
        if stats.total_vector_count > 0:
            perform_sample_queries(pc, INDEX_NAME, SAMPLE_QUERIES)
        else:
            print(f"\n⚠️  Index is empty - skipping sample queries")
            print(f"   Run the bulk ingestion script to add data:")
            print(f"   python scripts/bulk_ingest.py")
        
        print(f"\n✅ Pinecone validation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
