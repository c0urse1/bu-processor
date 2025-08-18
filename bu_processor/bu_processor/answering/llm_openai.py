from __future__ import annotations
from typing import List, Dict, Any
import re
from bu_processor.answering.models import AnswerResult, Citation
from bu_processor.answering.templates import SYSTEM_PROMPT, render_user_prompt

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)]$")  # match [1] or [1,2] at end of paragraph

class OpenAiAnswerer:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def answer(self, query: str, packed_context: str, sources_table: List[Dict[str, Any]]) -> AnswerResult:
        user_prompt = render_user_prompt(query, packed_context)
        
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=500
            )
            raw = resp.choices[0].message.content.strip()

            # Enforce paragraph-level citations if missing: append [1] to any paragraph lacking markers
            paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
            citations: List[Citation] = []
            repaired = []
            for i, p in enumerate(paras):
                m = _CITE_RE.search(p)
                indices: List[int] = []
                if m:
                    # parse indices like "2, 3"
                    indices = [int(x.strip()) for x in m.group(1).split(",") if x.strip().isdigit()]
                else:
                    indices = [1] if sources_table else []
                    p = p + (" [1]" if indices else "")
                repaired.append(p)

                for ci in indices:
                    if 1 <= ci <= len(sources_table):
                        st = sources_table[ci-1]
                        citations.append(Citation(paragraph_idx=i, chunk_id=st["chunk_id"], doc_id=st["doc_id"]))

            return AnswerResult(
                text="\n\n".join(repaired), 
                citations=citations, 
                sources_table=sources_table,
                trace={"model": self.model, "tokens_used": resp.usage.total_tokens}
            )
            
        except Exception as e:
            # Fallback error response
            return AnswerResult(
                text=f"I apologize, but I encountered an error while generating the answer: {str(e)}",
                citations=[],
                sources_table=sources_table,
                trace={"model": self.model, "error": str(e)}
            )
