from bu_processor.factories import make_embedder, make_index, make_store
from bu_processor.pipeline.upsert_pipeline import embed_and_index_chunks

def ingest_document(doc_title, doc_source, chunks):
    embedder = make_embedder()
    index = make_index()
    store = make_store()
    return embed_and_index_chunks(
        doc_title=doc_title,
        doc_source=doc_source,
        doc_meta={"ingest": "cli"},
        chunks=chunks,                    # [{"text": "...", "page": 3, "section": "1.2", "meta": {...}}, ...]
        embedder=embedder,
        index=index,
        store=store,
        namespace=None,
    )
