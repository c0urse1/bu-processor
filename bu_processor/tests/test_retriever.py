# tests/test_retriever.py
import numpy as np
import pytest

from bu_processor.embeddings.testing_backend import FakeDeterministicEmbeddings
from bu_processor.index.faiss_index import FaissIndex
from bu_processor.storage.sqlite_store import SQLiteStore
from bu_processor.pipeline.upsert_pipeline import embed_and_index_chunks

from bu_processor.retrieval.dense import DenseKnnRetriever
from bu_processor.retrieval.bm25 import Bm25Index
from bu_processor.retrieval.fusion import rrf_fuse, weighted_sum_fuse
from bu_processor.retrieval.hybrid import HybridRetriever

@pytest.fixture
def corpus_setup(tmp_path):
    """Fixture that sets up a corpus and returns store, embedder, index - with proper cleanup."""
    store = SQLiteStore(url=f"sqlite:///{tmp_path/'r.db'}")
    embedder = FakeDeterministicEmbeddings(dim=64)
    index = FaissIndex()
    
    chunks = [
        {"text": "Professional liability insurance covers financial losses.", "page": 1, "section": "Insurance", "meta": {"doc_type":"policy","date":"2024-06-01"}},
        {"text": "Pet insurance may cover vet visits.", "page": 2, "section": "Insurance", "meta": {"doc_type":"faq","date":"2024-05-12"}},
        {"text": "Cats are small domestic animals and love to sleep.", "page": 3, "section": "Pets", "meta": {"doc_type":"blog","date":"2023-12-10"}},
        {"text": "Corporate finance optimizes capital structure and funding.", "page": 4, "section": "Finance", "meta": {"doc_type":"guide","date":"2025-01-15"}},
        {"text": "Debt financing can be cheaper than equity financing.", "page": 5, "section": "Finance", "meta": {"doc_type":"guide","date":"2025-02-01"}},
        {"text": "Kittens are young cats; cats purr.", "page": 6, "section": "Pets", "meta": {"doc_type":"blog","date":"2024-02-02"}},
    ]
    embed_and_index_chunks(
        doc_title="Mini",
        doc_source="unit",
        doc_meta=None,
        chunks=chunks,
        embedder=embedder,
        index=index,
        store=store,
        namespace=None,
    )
    
    yield store, embedder, index
    
    # Cleanup
    store.close()

def test_dense_retrieval_basic(corpus_setup):
    store, embedder, index = corpus_setup
    dense = DenseKnnRetriever(embedder, index, store, namespace=None)
    hits = dense.retrieve("Which insurance covers financial loss?", top_k=3)
    assert len(hits) >= 1
    assert "insurance" in hits[0].text.lower()

def test_bm25_retrieval_basic(corpus_setup):
    store, embedder, index = corpus_setup

    bm25 = Bm25Index(store)
    bm25.build_from_store()
    hits = bm25.query("What is capital structure in corporate finance?", top_k=3)
    assert len(hits) >= 1
    assert hits[0].metadata.get("section") == "Finance"

def test_metadata_filters(corpus_setup):
    store, embedder, index = corpus_setup

    dense = DenseKnnRetriever(embedder, index, store, namespace=None)

    # Equality
    hits = dense.retrieve("Which insurance covers loss?", top_k=5, metadata_filter={"section":"Insurance"})
    assert all(h.metadata.get("section") == "Insurance" for h in hits)

    # IN filter
    hits2 = dense.retrieve("cats sleep a lot", top_k=5, metadata_filter={"section__in":["Pets","Finance"]})
    assert all(h.metadata.get("section") in ("Pets","Finance") for h in hits2)

    # Date range
    hits3 = dense.retrieve("financing", top_k=5, metadata_filter={"date_gte":"2025-01-01"})
    assert all(h.metadata.get("date") >= "2025-01-01" for h in hits3)

def test_fusion_rrf_and_weighted(corpus_setup):
    store, embedder, index = corpus_setup

    dense = DenseKnnRetriever(embedder, index, store, namespace=None)
    bm25 = Bm25Index(store); bm25.build_from_store()

    d_hits = dense.retrieve("equity vs debt financing", top_k=5)
    b_hits = bm25.query("equity vs debt financing", top_k=5)

    fused_rrf = rrf_fuse([d_hits, b_hits])
    fused_w = weighted_sum_fuse(d_hits, b_hits, alpha_dense=0.6, alpha_bm25=0.4)

    assert len(fused_rrf) >= 1 and len(fused_w) >= 1
    # The top finance id should appear near the top after fusion
    top_ids = [h.id for h in fused_rrf[:3]]
    assert any("finance" in h.metadata.get("section","").lower() or "finance" in h.text.lower() for h in fused_rrf[:3])

def test_hybrid_with_mmr(corpus_setup):
    store, embedder, index = corpus_setup

    dense = DenseKnnRetriever(embedder, index, store, namespace=None)
    bm25 = Bm25Index(store); bm25.build_from_store()

    hybrid = HybridRetriever(dense=dense, bm25=bm25, embedder=embedder,
                             fusion="rrf", use_mmr=True, mmr_lambda=0.65)

    hits = hybrid.retrieve("cats and corporate financing", final_top_k=3, top_k_dense=6, top_k_bm25=8)
    assert len(hits) == 3
    # Expect diversified topics across results (Pets + Finance in the top-3)
    secs = {h.metadata.get("section") for h in hits}
    assert "Pets" in secs and "Finance" in secs
