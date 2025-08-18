from __future__ import annotations
from typing import List, Dict, Any
import re
from bu_processor.answering.models import AnswerResult, Citation

def _pick_citations_for_paragraph(idx_start: int, used_source_count: int, max_per_para: int = 2) -> List[int]:
    """
    Simple deterministic policy: cite the first and (if available) second sources starting from idx_start.
    """
    cites = [idx_start]
    if used_source_count >= idx_start + 1:
        cites.append(idx_start + 1)
    return cites[:max_per_para]

def _format_cite_brackets(indices: List[int]) -> str:
    return "[" + ",".join(str(i) for i in sorted(set(indices))) + "]"

class RuleBasedAnswerer:
    """
    Deterministic synthesizer for CI & local runs (no LLM).
    - Builds 1–3 paragraphs from the top sources.
    - Copies/condenses a couple of high-signal sentences per source.
    - Appends numeric citation markers matching sources_table.
    """
    def answer(self, query: str, packed_context: str, sources_table: List[Dict[str, Any]]) -> AnswerResult:
        # Split the packed context by source sections (they are separated by blank lines)
        blocks = [b for b in re.split(r"\n\s*\n", packed_context) if b.strip()]
        # Grab up to 3 paragraph pieces
        paras = []
        for b in blocks[:3]:
            # take first 2 lines to reduce verbosity
            lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
            text = " ".join(lines[1:3]) if len(lines) > 1 else lines[0]  # skip the [i] header line
            paras.append(text)

        if not paras:
            return AnswerResult(text="Insufficient evidence to answer confidently.", sources_table=sources_table)

        # Build paragraphs with citations
        out_paras = []
        citations: List[Citation] = []
        for i, p in enumerate(paras):
            cite_indices = _pick_citations_for_paragraph(i+1, len(sources_table))
            out_paras.append(p + " " + _format_cite_brackets(cite_indices))
            for ci in cite_indices:
                st = sources_table[ci-1]
                citations.append(Citation(paragraph_idx=i, chunk_id=st["chunk_id"], doc_id=st["doc_id"]))

        return AnswerResult(text="\n\n".join(out_paras), citations=citations, sources_table=sources_table)
