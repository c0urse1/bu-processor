# scripts/pinecone_batch_test.py
import os
import time
import uuid
from dotenv import load_dotenv
from bu_processor.embeddings.embedder import Embedder
from bu_processor.integrations.pinecone_facade import make_pinecone_manager

# Load environment variables from .env file
load_dotenv()

def main():
    # Flag-Info nur als Hinweis (wir nutzen den Embedder wie konfiguriert)
    cache_enabled = os.getenv("ENABLE_EMBED_CACHE", "false").lower() in ("1","true","yes","on")
    print(f"🔁 Embedding-Cache: {'ON' if cache_enabled else 'OFF'}")

    # Eigener Namespace je Lauf, damit nichts kollidiert
    ns = (os.getenv("PINECONE_NAMESPACE", "bu-test") + "-" + uuid.uuid4().hex[:6])

    # Embedder & Pinecone
    embedder = Embedder()
    pc = make_pinecone_manager(
        index_name=os.getenv("PINECONE_INDEX_NAME", "bu-processor"),
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),
        cloud=os.getenv("PINECONE_CLOUD"),
        region=os.getenv("PINECONE_REGION"),
        namespace=ns
    )

    # Index-Guard & Erstellung
    idx_dim = pc.get_index_dimension()
    if idx_dim is not None and idx_dim != embedder.dimension:
        raise RuntimeError(f"Index-Dim {idx_dim} != Embedding-Dim {embedder.dimension}")
    pc.ensure_index(embedder.dimension)

    # 100 Test-“Chunks”
    N = 100
    texts = [f"BU-Testchunk {i} – Leistungsvoraussetzungen, Definitionen, Beispiele." for i in range(N)]
    ids   = [f"batch-{i}" for i in range(N)]
    metas = [{"doc_id":"batch-doc","i":i} for i in range(N)]

    print("🧮 Embeddings berechnen…")
    t0 = time.time()
    vecs = embedder.encode(texts, batch_size=64)
    t1 = time.time()
    print(f"✅ {N} Embeddings in {t1-t0:.2f}s")

    print("⬆️ Upsert in Pinecone…")
    pc.upsert_vectors(ids=ids, vectors=vecs, metadatas=metas)
    t2 = time.time()
    print(f"✅ Upsert abgeschlossen in {t2-t1:.2f}s (gesamt {t2-t0:.2f}s), namespace={ns}")

    print("🔍 Sanity-Query…")
    out = pc.query_by_text("Leistungsvoraussetzungen", embedder, top_k=3, include_metadata=True)
    matches = out.get("matches", [])
    print(f"🎯 Top-3 gefunden: {len(matches)}")
    for m in matches:
        print(f"  - id={m.get('id')} score={m.get('score'):.4f} meta={m.get('metadata')}")

    # Optionales Aufräumen: Comment aus, wenn du alles behalten willst
    pc.delete_by_document_id("batch-doc", namespace=ns)
    print("🧹 Cleanup: batch-doc gelöscht")

if __name__ == "__main__":
    main()
