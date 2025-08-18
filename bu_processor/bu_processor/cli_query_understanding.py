#!/usr/bin/env python3
"""
CLI example for query understanding with chat turns.

Usage:
    python cli_query_understanding.py

This demonstrates:
- Multi-turn chat conversation
- Query rewriting (heuristic/LLM)
- Query expansion (heuristic/LLM)
- Multi-query retrieval with RRF fusion
"""

from bu_processor.query.models import ChatTurn
from bu_processor.factories import make_query_pipeline, make_hybrid_retriever

def run(chat_turns):
    qp = make_query_pipeline()
    plan = qp.build_plan(chat_turns)
    print("Focused:", plan.focused_query)
    print("Expansions:", plan.expanded_queries)
    print("Trace:", plan.trace)
    print("All queries:", plan.all_queries)
    
    # Optional: retrieve across all queries (union + RRF)
    try:
        retriever = make_hybrid_retriever()
        hits = qp.retrieve_union(plan, retriever, top_k_per_query=5, final_top_k=5)
        print(f"\nRetrieved {len(hits)} results:")
        for i, h in enumerate(hits, 1):
            print(f"{i}. {h.score:.3f} [{h.metadata.get('section')}] {h.text[:90]}…")
    except Exception as e:
        print(f"Note: Retrieval skipped (no corpus): {e}")

if __name__ == "__main__":
    chat = [
        ChatTurn(role="user", content="Hi, quick question about insurance."),
        ChatTurn(role="assistant", content="Sure, what do you need?"),
        ChatTurn(role="user", content="Which insurance covers financial loss from professional mistakes?"),
    ]
    run(chat)
