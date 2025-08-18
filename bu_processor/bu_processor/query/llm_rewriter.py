from __future__ import annotations
from typing import List
from bu_processor.query.models import ChatTurn

PROMPT = """You condense multi-turn chat into ONE focused search query.
- No salutations or meta-talk.
- <= 20 words.
Messages:
{msgs}

Focused query:"""

class OpenAIRewriter:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai>=1.30")

    def rewrite(self, chat: List[ChatTurn]) -> str:
        msgs = "\n".join([f"{t.role.upper()}: {t.content}" for t in chat[-8:]])
        prompt = PROMPT.format(msgs=msgs)
        
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1
        )
        return resp.choices[0].message.content.strip()[:200]
