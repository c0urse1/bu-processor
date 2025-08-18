from __future__ import annotations
from typing import List
import re

class QueryAwareExtractiveSummarizer:
    """
    Deterministic extractive summarizer:
    - Sentence-split (naive regex).
    - Score sentences by query term coverage and proximity.
    - Return top sentences until token budget.
    """
    SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def _tokenize(self, s: str) -> List[str]:
        return re.findall(r"\w+", s.lower())

    def _score_sentence(self, query_terms: List[str], sent: str) -> float:
        toks = self._tokenize(sent)
        if not toks:
            return 0.0
        # coverage & density (hits per sentence length)
        hits = sum(1 for t in toks if t in query_terms)
        return hits / (len(toks) ** 0.5)

    def summarize(self, query: str, text: str, target_tokens: int = 160) -> str:
        if not text:
            return ""
        sents = re.split(self.SENT_SPLIT, text.strip())
        q_terms = set(self._tokenize(query))
        scored = [(self._score_sentence(list(q_terms), s), i, s) for i, s in enumerate(sents)]
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)  # score desc, earlier sentences tie-break

        sel = []
        tok_count = 0
        for sc, _, s in scored:
            if sc <= 0:
                continue
            toks = self._tokenize(s)
            if tok_count + len(toks) > target_tokens:
                continue
            sel.append(s)
            tok_count += len(toks)
            if len(sel) >= self.max_sentences:
                break

        # fallback: first sentence if nothing scored
        if not sel and sents:
            sel = [sents[0]]
        return " ".join(sel)
