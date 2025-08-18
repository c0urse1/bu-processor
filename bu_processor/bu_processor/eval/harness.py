from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import asdict

from bu_processor.embeddings.testing_backend import FakeDeterministicEmbeddings
from bu_processor.index.faiss_index import FaissIndex
from bu_processor.storage.sqlite_store import SQLiteStore
from bu_processor.pipeline.upsert_pipeline import embed_and_index_chunks
from bu_processor.retrieval.dense import DenseKnnRetriever
from bu_processor.retrieval.bm25 import Bm25Index
from bu_processor.retrieval.hybrid import HybridRetriever
from bu_processor.rerank.testing_reranker import HeuristicOverlapReranker
from bu_processor.summarize.query_aware_extractive import QueryAwareExtractiveSummarizer
from bu_processor.answering.rule_based import RuleBasedAnswerer
from bu_processor.answering.synthesize import synthesize_answer
from bu_processor.eval.metrics import hit_at_k, reciprocal_rank, aggregate_metrics, citation_accuracy, faithfulness_keywords

def build_default_stack(sqlite_url: str) -> Tuple[SQLiteStore, FakeDeterministicEmbeddings, FaissIndex, HybridRetriever]:
    store = SQLiteStore(url=sqlite_url)
    embedder = FakeDeterministicEmbeddings(dim=64)
    index = FaissIndex()
    dense = DenseKnnRetriever(embedder, index, store)
    bm25  = Bm25Index(store); bm25.build_from_store()
    hybrid = HybridRetriever(
        dense=dense, bm25=bm25, embedder=embedder,
        fusion="rrf", use_mmr=True, mmr_lambda=0.65,
        reranker=HeuristicOverlapReranker(),
        summarizer=QueryAwareExtractiveSummarizer(),
    )
    return store, embedder, index, hybrid

def ingest_corpus(store: SQLiteStore, embedder, index, docs: List[Dict[str, Any]]):
    """
    docs: [{"title":..., "source":..., "chunks":[{"text":..., "page":int?, "section":str?, "meta":{...}}, ...]}, ...]
    """
    for d in docs:
        embed_and_index_chunks(
            doc_title=d.get("title"),
            doc_source=d.get("source", "eval"),
            doc_meta=d.get("meta"),
            chunks=d["chunks"],
            embedder=embedder,
            index=index,
            store=store,
            namespace=None,
        )

def run_eval(
    sqlite_url: str,
    corpus_docs: List[Dict[str, Any]],
    golden_set: List[Dict[str, Any]],
    top_k_retrieve: int = 5,
) -> Dict[str, Any]:
    store, embedder, index, retriever = build_default_stack(sqlite_url)
    ingest_corpus(store, embedder, index, corpus_docs)

    rows: List[Dict[str, Any]] = []
    for item in golden_set:
        q = item["query"]
        gold_doc_ids = item.get("gold_doc_ids", [])
        gold_chunk_ids = set(item.get("gold_chunk_ids", []))
        answer_keywords = item.get("answer_keywords", [])

        # Retrieve and synthesize (deterministic)
        hits = retriever.retrieve(q, final_top_k=top_k_retrieve)
        ret_ids = [h.id for h in hits]
        ret_doc_ids = [h.metadata.get("doc_id") for h in hits]

        # Metrics: retrieval
        row = {
            "query": q,
            "hit@1": hit_at_k(ret_doc_ids, gold_doc_ids, 1),
            "hit@3": hit_at_k(ret_doc_ids, gold_doc_ids, 3),
            "hit@5": hit_at_k(ret_doc_ids, gold_doc_ids, 5),
            "mrr": reciprocal_rank(ret_doc_ids, gold_doc_ids),
        }

        # Synthesis
        ans = synthesize_answer(query=q, hits=hits, answerer=RuleBasedAnswerer(), token_budget=600)
        # Citation accuracy
        row["citation_acc"] = citation_accuracy(ans.text, ans.sources_table)

        # Faithfulness (keyword heuristic)
        cited_texts = []
        for c in ans.citations:
            # find this chunk's text back (sources_table already contains ids; we just pick the id's text from hits)
            for h in hits:
                if h.id == c.chunk_id:
                    cited_texts.append(h.text)
                    break
        row["faithfulness"] = faithfulness_keywords(ans.text, cited_texts, answer_keywords)

        rows.append(row)

    agg = aggregate_metrics(rows)
    return {"rows": rows, "aggregate": agg}

def save_report(report: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
