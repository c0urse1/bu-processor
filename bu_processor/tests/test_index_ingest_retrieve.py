from bu_processor.embeddings.testing_backend import FakeDeterministicEmbeddings
from bu_processor.index.faiss_index import FaissIndex
from bu_processor.storage.sqlite_store import SQLiteStore
from bu_processor.pipeline.upsert_pipeline import embed_and_index_chunks
from bu_processor.pipeline.retrieve import retrieve_similar

def test_ingest_and_retrieve_smoke(tmp_path):
    store = SQLiteStore(url=f"sqlite:///{tmp_path/'test.db'}")
    embedder = FakeDeterministicEmbeddings(dim=64)
    index = FaissIndex()

    chunks = [
        {"text": "Professional liability insurance covers financial losses." , "page": 1, "section": "Intro"},
        {"text": "Cats are small domestic animals and love to sleep."       , "page": 2, "section": "Pets"},
        {"text": "Corporate finance optimizes capital structure and funding.", "page": 3, "section": "Finance"},
    ]

    res = embed_and_index_chunks(
        doc_title="Sample Doc",
        doc_source="unittest",
        doc_meta={"tenant": "acme"},
        chunks=chunks,
        embedder=embedder,
        index=index,
        store=store,
    )

    assert "doc_id" in res and len(res["chunk_ids"]) == 3

    hits = retrieve_similar(
        query="Which insurance covers financial loss?",
        embedder=embedder,
        index=index,
        store=store,
        top_k=2,
    )
    assert len(hits) >= 1
    assert "insurance" in hits[0]["text"].lower()
    assert "chunk_id" in hits[0]["metadata"]
    assert "doc_id" in hits[0]["metadata"]


# These tests run without FAISS if the package is missing (numpy fallback kicks in), without OpenAI, and without Pinecone.
