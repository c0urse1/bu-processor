from __future__ import annotations
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from bu_processor.query.models import ChatTurn, QueryPlan
from bu_processor.ports import QueryRewriter, QueryExpander
from bu_processor.retrieval.models import RetrievalHit
from bu_processor.retrieval.fusion import rrf_fuse

class QueryPipeline:
    def __init__(self,
                 rewriter: Optional[QueryRewriter],
                 expander: Optional[QueryExpander],
                 enable_rewrite: bool = True,
                 enable_expand: bool = True,
                 expansions_k: int = 2):
        self.rewriter = rewriter
        self.expander = expander
        self.enable_rewrite = enable_rewrite
        self.enable_expand = enable_expand
        self.expansions_k = max(0, expansions_k)

    # -------- Stage 1: Build plan (no retrieval) --------
    def build_plan(self, chat: List[ChatTurn]) -> QueryPlan:
        trace: Dict[str, Any] = {}
        # Focused query
        if self.enable_rewrite and self.rewriter is not None:
            focused = self.rewriter.rewrite(chat).strip()
            trace["rewriter"] = type(self.rewriter).__name__
        else:
            # fallback: last user turn or empty
            focused = ""
            for t in reversed(chat):
                if t.role == "user":
                    focused = t.content.strip()
                    break
            trace["rewriter"] = "fallback_last_user"
        focused = " ".join(focused.split())[:200]

        # Expansions
        expanded: List[str] = []
        if self.enable_expand and self.expander is not None and self.expansions_k > 0 and focused:
            expanded = self.expander.expand(focused, num=self.expansions_k)
            trace["expander"] = type(self.expander).__name__
        else:
            trace["expander"] = "disabled_or_none"

        return QueryPlan(focused_query=focused, expanded_queries=expanded, trace=trace)

    # -------- Stage 2 (optional): Multi-query retrieval + fusion --------
    def retrieve_union(self,
                       plan: QueryPlan,
                       retriever,                 # your Dense/Hybrid retriever -> .retrieve(query,...)
                       top_k_per_query: int = 5,
                       final_top_k: int = 5,
                       metadata_filter: Optional[Dict[str, Any]] = None) -> List[RetrievalHit]:
        """
        Runs retrieval for each query in the plan, fuses with RRF, returns top-N hits.
        """
        per_query_results: List[List[RetrievalHit]] = []
        for q in plan.all_queries:
            hits = retriever.retrieve(q, final_top_k=top_k_per_query, metadata_filter=metadata_filter)
            per_query_results.append(hits)

        fused = rrf_fuse(per_query_results)
        return fused[:final_top_k]
