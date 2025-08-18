from typing import List, Optional
from bu_processor.retrieval.models import RetrievalHit
from bu_processor.ports import Reranker, QueryAwareSummarizer

def rerank_and_summarize(query: str,
                         hits: List[RetrievalHit],
                         reranker: Optional[Reranker],
                         summarizer: Optional[QueryAwareSummarizer],
                         summary_tokens: int = 160,
                         top_k: Optional[int] = None) -> List[RetrievalHit]:
    """
    Post-retrieval processing: rerank hits and add query-aware summaries.
    Can be used as alternative to integrating directly into retrievers.
    """
    out = hits
    if reranker is not None:
        out = reranker.rerank(query, out, top_k=top_k)
    elif top_k is not None:
        out = out[:top_k]

    if summarizer is not None:
        for h in out:
            h.metadata = dict(h.metadata)
            h.metadata["summary"] = summarizer.summarize(query, h.text, target_tokens=summary_tokens)
    return out
