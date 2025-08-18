# bu_processor/cli_query.py
from bu_processor.factories import make_hybrid_retriever

def ask(query: str, top_k: int = 5, flt=None):
    retr = make_hybrid_retriever()
    hits = retr.retrieve(query, final_top_k=top_k, metadata_filter=flt)
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h.score:.3f}  [{h.metadata.get('section')}]  {h.text[:100]}…  (chunk_id={h.id})")
