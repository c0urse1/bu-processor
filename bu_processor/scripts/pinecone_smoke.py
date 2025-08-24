# scripts/pinecone_smoke.py
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bu_processor.embeddings.embedder import Embedder
from bu_processor.integrations.pinecone_facade import make_pinecone_manager
from bu_processor.core.flags import ENABLE_ENHANCED_PINECONE, ENABLE_EMBED_CACHE, ENABLE_RERANK

def main():
    print("🔍 Starting Pinecone Smoke Test...")
    print("=" * 50)
    
    # Show feature flags status
    print("🚩 Feature Flags Status:")
    print(f"   ENABLE_ENHANCED_PINECONE: {ENABLE_ENHANCED_PINECONE}")
    print(f"   ENABLE_EMBED_CACHE: {ENABLE_EMBED_CACHE}")
    print(f"   ENABLE_RERANK: {ENABLE_RERANK}")
    print()
    
    # Initialize components using standardized wiring
    print("1. Initializing Embedder...")
    embedder = Embedder()
    print(f"   Model: {embedder.model_name}")
    print(f"   Dimension: {embedder.dimension}")
    
    print("2. Initializing PineconeManager using standardized wiring...")
    pc = make_pinecone_manager(
        index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),      # v2
        cloud=os.getenv("PINECONE_CLOUD"),          # v3
        region=os.getenv("PINECONE_REGION"),        # v3
        namespace=os.getenv("PINECONE_NAMESPACE")
    )
    print(f"   Implementation: {pc.implementation_type}")
    print(f"   Enhanced: {pc.is_enhanced}")
    
    print("3. Ensuring index exists...")
    pc.ensure_index(dimension=embedder.dimension)
    
    print("4. Preparing test data...")
    texts = [
        "BU leistet bei dauerhafter Berufsunfähigkeit.", 
        "Kündigungsfrist & Nachversicherung.",
        "Machine learning algorithms for document classification."
    ]
    ids = ["smoke-test-1", "smoke-test-2", "smoke-test-3"]
    metas = [
        {"doc_id": "smoke-doc", "source": "smoke", "type": "test", "text": texts[0]}, 
        {"doc_id": "smoke-doc", "source": "smoke", "type": "test", "text": texts[1]},
        {"doc_id": "smoke-doc", "source": "smoke", "type": "test", "text": texts[2]}
    ]
    
    print("5. Encoding texts...")
    import time
    start_time = time.time()
    vecs = embedder.encode(texts)
    encoding_time = time.time() - start_time
    print(f"   Encoded {len(vecs)} vectors with dimension {len(vecs[0])} in {encoding_time:.3f}s")
    
    print("6. Upserting vectors to Pinecone with quality gates...")
    result = pc.upsert_vectors(
        ids=ids, 
        vectors=vecs, 
        metadatas=metas,
        embedder=embedder  # This triggers quality gates
    )
    print(f"   Upsert result: {result}")
    print("   ✅ Quality gates passed")
    
    print("7. Testing query capabilities...")
    query_text = "Berufsunfähigkeitsversicherung"
    print(f"   Query: '{query_text}'")
    
    # Test query with potential reranking
    query_result = pc.query_by_text(
        text=query_text,
        embedder=embedder,
        top_k=3,
        include_metadata=True
    )
    
    print(f"   Found {len(query_result.get('matches', []))} matches")
    for i, match in enumerate(query_result.get('matches', [])[:2]):
        score = match.get('score', 0)
        text = match.get('metadata', {}).get('text', 'No text')[:50]
        print(f"   {i+1}. Score: {score:.3f} - {text}...")
    
    if query_result.get('reranked'):
        print("   ✅ Results were reranked with cross-encoder")
    
    print()
    print("🎯 Smoke Test Results:")
    print("   ✅ Embedder working")
    print("   ✅ Standardized wiring working") 
    print("   ✅ Quality gates functioning")
    print("   ✅ Upsert operation successful")
    print("   ✅ Query operation successful")
    print(f"   ✅ Implementation: {pc.implementation_type}")
    print()
    
    if ENABLE_EMBED_CACHE:
        print("🔄 Testing cache performance...")
        start_time = time.time()
        vecs_cached = embedder.encode(texts)
        cached_time = time.time() - start_time
        print(f"   Second encoding took {cached_time:.3f}s vs {encoding_time:.3f}s first time")
        if cached_time < encoding_time * 0.5:
            print("   ✅ Cache appears to be working (significant speedup)")
        else:
            print("   ⚠️  Cache may not be working optimally")
    
    print("🚀 Smoke test completed successfully!")
    
    print("7. Testing query...")
    query_result = pc.query_by_text("Wann zahlt BU?", embedder, top_k=2)
    print(f"   Query result: {query_result}")
    
    print("✅ Smoke test completed successfully!")

if __name__ == "__main__":
    main()
