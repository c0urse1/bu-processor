from __future__ import annotations
from typing import List
import re
from bu_processor.query.models import ChatTurn

_STOP = {
    "please","plz","could","would","should","tell","me","about","the","a","an",
    "hi","hello","thanks","thank","you","help","need","how","what","which","that","this"
}

class HeuristicRewriter:
    """
    Deterministic, no-network:
    - Pick last user message.
    - Strip greetings/fluff.
    - If there's punctuation-separated clauses, keep the last (most specific).
    - Keep salient tokens in order and trim length.
    """
    _SPLIT = re.compile(r"[.?!;\n]+")
    _WORD  = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\-]+")

    def rewrite(self, chat: List[ChatTurn]) -> str:
        last = ""
        for turn in reversed(chat):
            if turn.role == "user":
                last = turn.content.strip()
                break
        if not last:
            return ""

        last = re.sub(r"^(hi|hello|hey)[,!\s]*", "", last, flags=re.IGNORECASE).strip()
        clauses = [c.strip() for c in self._SPLIT.split(last) if c.strip()]
        if clauses:
            last = clauses[-1]

        toks = [t.lower() for t in self._WORD.findall(last)]
        toks = [t for t in toks if t not in _STOP and len(t) > 1]

        seen, ordered = set(), []
        for t in toks:
            if t not in seen:
                ordered.append(t); seen.add(t)

        if not ordered:
            return last[:200]
        return " ".join(ordered[:24])
