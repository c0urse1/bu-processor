# tests/test_embeddings_index.py
import os
import math
import json
import tempfile
import numpy as np
import pytest

# Embedders
from bu_processor.embeddings.testing_backend import FakeDeterministicEmbeddings

# Optional real embedders (guarded)
SBERT_AVAILABLE = True
try:
    from bu_processor.embeddings.sbert_backend import SbertEmbeddings  # noqa: F401
except Exception:
    SBERT_AVAILABLE = False

OPENAI_AVAILABLE = True
try:
    # import is light; actual network calls are guarded by env in tests
    from bu_processor.embeddings.openai_backend import OpenAIEmbeddings  # noqa: F401
except Exception:
    OPENAI_AVAILABLE = False

# Indexes
from bu_processor.index.faiss_index import FaissIndex

PINECONE_AVAILABLE = True
try:
    from bu_processor.index.pinecone_index import PineconeIndex  # noqa: F401
except Exception:
    PINECONE_AVAILABLE = False

# Store + pipeline
from bu_processor.storage.sqlite_store import SQLiteStore
from bu_processor.pipeline.upsert_pipeline import embed_and_index_chunks
from bu_processor.pipeline.retrieve import retrieve_similar


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture(scope="function")
def tmp_db_url(tmp_path):
    return f"sqlite:///{tmp_path/'test.db'}"


@pytest.fixture(scope="function")
def store(tmp_db_url):
    return SQLiteStore(url=tmp_db_url)


@pytest.fixture(scope="function")
def fake_embedder():
    # small dim keeps tests fast
    return FakeDeterministicEmbeddings(dim=64)


@pytest.fixture(scope="function")
def faiss_index():
    # Works with faiss if installed; otherwise uses numpy fallback in adapter
    return FaissIndex()


# -------------------------
# Unit tests: Embedder
# -------------------------

def test_fake_embedder_is_deterministic(fake_embedder):
    texts = ["Cats are lovely.", "Cats are lovely.", "Finance optimizes capital."]
    vecs = fake_embedder.encode(texts)
    assert vecs.shape == (3, 64)

    # same text -> identical vectors
    assert np.allclose(vecs[0], vecs[1])

    # different text -> not identical (very likely)
    assert not np.allclose(vecs[0], vecs[2])

    # normalized vectors (L2 ~ 1.0)
    for i in range(3):
        norm = np.linalg.norm(vecs[i])
        assert math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5)


# -------------------------
# Unit tests: FAISS index
# -------------------------

def test_faiss_index_create_upsert_query(fake_embedder, faiss_index):
    texts = [
        "Professional liability insurance covers financial losses.",
        "Cats are small domestic animals.",
        "Corporate finance optimizes capital structure and funding.",
    ]
    vecs = fake_embedder.encode(texts)
    dim = vecs.shape[1]

    # create index
    faiss_index.create(dim=dim, metric="cosine", namespace="test")

    ids = ["c_ins", "c_cat", "c_fin"]
    metadata = [
        {"section": "Insurance", "page": 1},
        {"section": "Pets", "page": 2},
        {"section": "Finance", "page": 3},
    ]

    faiss_index.upsert(ids=ids, vectors=vecs, metadata=metadata, namespace="test")

    # query for finance-like question
    q = "What is capital structure in corporate finance?"
    qv = fake_embedder.encode([q])[0]
    hits = faiss_index.query(qv, top_k=3, namespace="test")

    assert len(hits) >= 1
    # expect the finance chunk near the top
    top_ids = [h["id"] for h in hits]
    assert "c_fin" in top_ids[:2]


def test_faiss_index_metadata_filter(fake_embedder, faiss_index):
    texts = [
        "Health insurance includes inpatient and outpatient coverage.",
        "Corporate finance covers debt and equity financing.",
        "Pet insurance may include vet visits.",
    ]
    ids = ["a", "b", "c"]
    metas = [
        {"section": "Insurance", "topic": "health"},
        {"section": "Finance", "topic": "corporate"},
        {"section": "Insurance", "topic": "pets"},
    ]
    vecs = fake_embedder.encode(texts)
    faiss_index.create(dim=vecs.shape[1], metric="cosine", namespace=None)
    faiss_index.upsert(ids=ids, vectors=vecs, metadata=metas, namespace=None)

    q = "How do companies balance debt and equity?"
    qv = fake_embedder.encode([q])[0]

    # Filter for Insurance should exclude the finance hit
    hits_ins = faiss_index.query(qv, top_k=3, namespace=None, metadata_filter={"section": "Insurance"})
    assert all(h["metadata"].get("section") == "Insurance" for h in hits_ins)

    # Filter for Finance should return only the corporate finance one
    hits_fin = faiss_index.query(qv, top_k=3, namespace=None, metadata_filter={"section": "Finance"})
    assert len(hits_fin) >= 1
    assert hits_fin[0]["metadata"].get("topic") == "corporate"


# -------------------------
# Integration: upsert + retrieve
# -------------------------

def test_embed_and_index_chunks_then_retrieve(store, fake_embedder, faiss_index):
    chunks = [
        {"text": "Professional liability insurance covers financial losses.", "page": 1, "section": "Intro"},
        {"text": "Cats are small domestic animals and love to sleep.", "page": 2, "section": "Pets"},
        {"text": "Corporate finance optimizes capital structure and funding.", "page": 3, "section": "Finance"},
    ]

    res = embed_and_index_chunks(
        doc_title="Sample Doc",
        doc_source="unit-test",
        doc_meta={"tenant": "acme"},
        chunks=chunks,
        embedder=fake_embedder,
        index=faiss_index,
        store=store,
        namespace="ns1",
    )

    assert "doc_id" in res and isinstance(res["doc_id"], str)
    assert len(res["chunk_ids"]) == 3
    assert res["dim"] == 64

    # retrieval
    hits = retrieve_similar(
        query="Which insurance covers financial loss?",
        embedder=fake_embedder,
        index=faiss_index,
        store=store,
        top_k=2,
        namespace="ns1",
    )
    assert len(hits) >= 1
    assert "financial" in hits[0]["text"].lower()
    # sanity: metadata is present
    assert "doc_id" in hits[0]["metadata"]
    assert "chunk_id" in hits[0]["metadata"]


def test_retrieve_with_metadata_filter(store, fake_embedder, faiss_index):
    chunks = [
        {"text": "BM25 is a lexical ranking function used by search engines.", "page": 1, "section": "IR"},
        {"text": "Debt financing can be cheaper than equity financing.", "page": 2, "section": "Finance"},
        {"text": "Cats purr and sleep a lot.", "page": 3, "section": "Pets"},
    ]
    embed_and_index_chunks(
        doc_title="Mix Doc",
        doc_source="unit-test",
        doc_meta=None,
        chunks=chunks,
        embedder=fake_embedder,
        index=faiss_index,
        store=store,
        namespace="ns2",
    )

    hits_all = retrieve_similar(
        query="How does equity compare to debt for companies?",
        embedder=fake_embedder,
        index=faiss_index,
        store=store,
        top_k=3,
        namespace="ns2",
    )
    assert len(hits_all) >= 1

    hits_fin = retrieve_similar(
        query="How does equity compare to debt for companies?",
        embedder=fake_embedder,
        index=faiss_index,
        store=store,
        top_k=3,
        namespace="ns2",
        metadata_filter={"section": "Finance"},
    )
    assert len(hits_fin) >= 1
    assert hits_fin[0]["metadata"].get("section") == "Finance"


# -------------------------
# Optional slow tests (real backends)
# -------------------------

@pytest.mark.slow
@pytest.mark.skipif(not SBERT_AVAILABLE or not os.getenv("RUN_REAL_SEMANTIC"),
                    reason="Set RUN_REAL_SEMANTIC=1 and install sentence-transformers to run")
def test_sbert_embedding_roundtrip(tmp_db_url):
    store = SQLiteStore(url=tmp_db_url)
    embedder = SbertEmbeddings(model_name=os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    index = FaissIndex()

    chunks = [
        {"text": "Corporate finance balances risk and return.", "page": 1},
        {"text": "Kittens are young cats.", "page": 2},
    ]
    res = embed_and_index_chunks(
        doc_title="SBERT Doc",
        doc_source="slow",
        doc_meta=None,
        chunks=chunks,
        embedder=embedder,
        index=index,
        store=store,
        namespace=None,
    )
    assert len(res["chunk_ids"]) == 2

    hits = retrieve_similar(
        query="What do companies consider when financing projects?",
        embedder=embedder,
        index=index,
        store=store,
        top_k=2,
    )
    assert len(hits) >= 1
    assert "finance" in hits[0]["text"].lower()


@pytest.mark.slow
@pytest.mark.skipif(
    not (OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY") and os.getenv("RUN_OPENAI_EMBEDDINGS")),
    reason="Set RUN_OPENAI_EMBEDDINGS=1 and OPENAI_API_KEY to run"
)
def test_openai_embedding_roundtrip(tmp_db_url):
    from bu_processor.embeddings.openai_backend import OpenAIEmbeddings
    store = SQLiteStore(url=tmp_db_url)
    embedder = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"],
                                model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    index = FaissIndex()

    chunks = [{"text": "Professional liability insurance protects against financial loss."}]
    res = embed_and_index_chunks(
        doc_title="OpenAI Doc",
        doc_source="slow",
        doc_meta=None,
        chunks=chunks,
        embedder=embedder,
        index=index,
        store=store,
    )
    assert len(res["chunk_ids"]) == 1

    hits = retrieve_similar(
        query="Which insurance covers financial loss from mistakes?",
        embedder=embedder,
        index=index,
        store=store,
        top_k=1,
    )
    assert len(hits) == 1
    assert "insurance" in hits[0]["text"].lower()


@pytest.mark.slow
@pytest.mark.skipif(
    not (PINECONE_AVAILABLE and os.getenv("PINECONE_API_KEY") and os.getenv("RUN_PINECONE_TESTS")),
    reason="Set RUN_PINECONE_TESTS=1 and PINECONE_API_KEY to run"
)
def test_pinecone_adapter_roundtrip(monkeypatch):
    # NOTE: this test exercises adapter API; it will create a temporary serverless index
    from bu_processor.index.pinecone_index import PineconeIndex
    from bu_processor.embeddings.testing_backend import FakeDeterministicEmbeddings

    api_key = os.environ["PINECONE_API_KEY"]
    index_name = os.getenv("PINECONE_INDEX", "bu-index-ci")

    embedder = FakeDeterministicEmbeddings(dim=64)
    index = PineconeIndex(api_key=api_key, index_name=index_name)
    vecs = embedder.encode(["Debt financing can be cheaper than equity."])
    index.create(dim=vecs.shape[1], metric="cosine", namespace="ci-ns")

    ids = ["pc1"]
    meta = [{"section": "Finance"}]
    index.upsert(ids=ids, vectors=vecs, metadata=meta, namespace="ci-ns")

    q = "Compare equity vs debt for companies."
    qv = embedder.encode([q])[0]
    hits = index.query(qv, top_k=3, namespace="ci-ns")
    assert len(hits) >= 1
    assert hits[0]["id"] == "pc1"


# How to run
#
# Fast, deterministic tests (no network):
# pytest -q
#
# Optional slow SBERT test:
# RUN_REAL_SEMANTIC=1 pytest -q -m slow
#
# Optional OpenAI embeddings:
# export OPENAI_API_KEY=sk-...
# RUN_OPENAI_EMBEDDINGS=1 pytest -q -m slow
#
# Optional Pinecone adapter:
# export PINECONE_API_KEY=pc-...
# export PINECONE_INDEX=bu-index-ci   # optional
# RUN_PINECONE_TESTS=1 pytest -q -m slow
#
# What this covers:
# - Deterministic embedding behavior (no downloads).
# - FAISS (or numpy fallback) index create/upsert/query.
# - Full ingest → embed → index → SQLite persist → retrieve → text+metadata for citations.
# - Metadata filtering.
# - Optional real-backend smoke tests gated by env flags.
#
# If your module paths differ, adjust the imports at the top; the rest can stay as-is.
