#!/usr/bin/env python3
"""
Demo CLI for the new simplified Pinecone integration.
This demonstrates the exact wiring described in the requirements.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bu_processor.embeddings.embedder import Embedder
from bu_processor.integrations.pinecone_facade import make_pinecone_manager

def demo_simplified_integration():
    """Demonstrate the exact wiring pattern from the requirements."""
    
    print("🚀 Demo: Simplified Pinecone Integration")
    print("=" * 50)
    
    # 1. Initialize components exactly as specified
    print("1. Initializing components...")
    
    embedder = Embedder()
    pc = make_pinecone_manager(
        index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),      # v2
        cloud=os.getenv("PINECONE_CLOUD"),          # v3
        region=os.getenv("PINECONE_REGION"),        # v3
        metric="cosine",
        namespace=os.getenv("PINECONE_NAMESPACE")   # optional
    )
    
    print(f"   ✅ Embedder: {embedder.model_name} (dim: {embedder.dimension})")
    print(f"   ✅ PineconeManager: {pc.index_name}")
    
    # 2. Index sicherstellen (Dimension passend zum Embedder!)
    print("2. Ensuring index exists...")
    pc.ensure_index(dimension=embedder.dimension)
    print(f"   ✅ Index ensured with dimension {embedder.dimension}")
    
    # 3. Prepare test data
    print("3. Preparing test data...")
    texts = [
        "BU leistet bei dauerhafter Berufsunfähigkeit ab 50% Grad.",
        "Kündigungsfrist beträgt 3 Monate zum Versicherungsende.",
        "Nachversicherungsgarantie ohne Gesundheitsprüfung verfügbar."
    ]
    ids = ["bu-doc-1", "bu-doc-2", "bu-doc-3"]
    metadatas = [
        {"doc_id": "bu-bedingungen", "section": "leistungen", "page": 1},
        {"doc_id": "bu-bedingungen", "section": "kuendigung", "page": 5},
        {"doc_id": "bu-bedingungen", "section": "nachversicherung", "page": 7}
    ]
    
    print(f"   ✅ Prepared {len(texts)} texts for indexing")
    
    # 4. Generate embeddings
    print("4. Generating embeddings...")
    vectors = embedder.encode(texts)
    print(f"   ✅ Generated {len(vectors)} vectors")
    
    # 5. Upsert to Pinecone
    print("5. Upserting to Pinecone...")
    result = pc.upsert_vectors(ids=ids, vectors=vectors, metadatas=metadatas)
    print(f"   ✅ Upsert result: {result}")
    
    # 6. Test query
    print("6. Testing query...")
    query_result = pc.query_by_text(
        "Was ist BU-Leistung?", 
        embedder, 
        top_k=5, 
        include_metadata=True
    )
    
    print(f"   ✅ Query completed")
    print(f"   📊 Found {len(query_result.get('matches', []))} matches:")
    
    for i, match in enumerate(query_result.get('matches', []), 1):
        score = match.get('score', 0)
        match_id = match.get('id', 'N/A')
        metadata = match.get('metadata', {})
        print(f"      {i}. {match_id} (score: {score:.3f})")
        print(f"         Section: {metadata.get('section', 'N/A')}")
        print(f"         Page: {metadata.get('page', 'N/A')}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nThis demonstrates the exact integration pattern specified:")
    print("- ✅ Embedder initialization from ENV/Config")
    print("- ✅ PineconeManager with v2/v3 compatibility")
    print("- ✅ Automatic index dimension management")
    print("- ✅ Simple upsert_vectors API")
    print("- ✅ Text-based query interface")

if __name__ == "__main__":
    try:
        demo_simplified_integration()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
