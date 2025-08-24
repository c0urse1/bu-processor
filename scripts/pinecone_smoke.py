#!/usr/bin/env python3
"""
🧪 PINECONE SMOKE TEST
===================
Mini-Smoke-Test für Pinecone Integration ohne API/Worker Komponenten.
Testet die grundlegende Funktionalität der vereinfachten Pinecone-Integration.
"""

# scripts/pinecone_smoke.py (aus Repo-Root starten)
import os
from bu_processor.embeddings.embedder import Embedder
from bu_processor.integrations.pinecone_manager import PineconeManager

def main():
    print("🧪 PINECONE SMOKE TEST")
    print("=" * 50)
    
    try:
        print("🔧 Initializing Embedder...")
        embedder = Embedder()  # verwendet ENV EMBEDDING_MODEL oder Default (mpnet-base-v2, 768D)
        print(f"✅ Embedder initialized (dimension: {embedder.dimension})")
        
        print("\n🔧 Initializing PineconeManager...")
        pc = PineconeManager(
            index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV"),   # v2
            cloud=os.getenv("PINECONE_CLOUD"),       # v3
            region=os.getenv("PINECONE_REGION"),     # v3
            namespace=os.getenv("PINECONE_NAMESPACE")
        )
        print("✅ PineconeManager initialized")
        
        print(f"\n🔧 Ensuring index exists (dimension: {embedder.dimension})...")
        pc.ensure_index(dimension=embedder.dimension)
        print("✅ Index ready")

        print("\n📝 Preparing test data...")
        texts = ["BU leistet bei dauerhafter Berufsunfähigkeit.", "Kündigungsfrist & Nachversicherung."]
        ids   = ["test-1", "test-2"]
        metas = [{"doc_id": "smoke-doc"}, {"doc_id": "smoke-doc"}]
        
        print("🧮 Generating embeddings...")
        vecs  = embedder.encode(texts)
        print(f"✅ Generated {len(vecs)} embeddings")

        print("\n⬆️ Upserting vectors to Pinecone...")
        pc.upsert_vectors(ids=ids, vectors=vecs, metadatas=metas)
        print("✅ Vectors upserted successfully")
        
        print("\n🔍 Testing query...")
        out = pc.query_by_text("Wann zahlt BU?", embedder, top_k=2)
        
        print("\n🎯 QUERY RESULT:")
        print("-" * 30)
        print(f"Query: 'Wann zahlt BU?'")
        print(f"Results found: {len(out.get('matches', []))}")
        
        for i, match in enumerate(out.get("matches", []), 1):
            print(f"\nResult {i}:")
            print(f"  ID: {match.get('id', 'N/A')}")
            print(f"  Score: {match.get('score', 'N/A'):.4f}")
            print(f"  Metadata: {match.get('metadata', {})}")
        
        print("\n🎉 SMOKE TEST COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
