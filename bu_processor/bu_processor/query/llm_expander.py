from __future__ import annotations
from typing import List

PROMPT = """Generate {k} diverse paraphrases for the search query.
- Keep each <= 16 words.
- Be meaning-preserving.
Query: "{q}"

Paraphrases:"""

class OpenAIExpander:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai>=1.30")

    def expand(self, focused_query: str, num: int = 2) -> List[str]:
        k = max(2, num)
        prompt = PROMPT.format(k=k, q=focused_query.strip())
        
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        
        lines = [ln.strip("-• ").strip() for ln in resp.choices[0].message.content.splitlines()]
        out = [x for x in lines if x]
        return out[:k]
