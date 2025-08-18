SYSTEM_PROMPT = """You are a careful, citation-first assistant. Answer ONLY using the provided sources.
If you are not sure, say you do not have enough information. Be concise and specific.
Rules:
- Do not invent facts.
- Use neutral tone.
- Add a citation marker at the END of EACH PARAGRAPH like [1], [2,3].
- If multiple sources support a paragraph, cite them comma-separated in ascending order.
"""

USER_PROMPT_TEMPLATE = """Question:
{query}

Sources (numbered):
{sources_block}

Instructions:
- Use the sources above only.
- Structure your answer into clear paragraphs. Each paragraph MUST end with citation brackets.
- If the sources conflict or are insufficient, say: "Insufficient evidence to answer confidently." and briefly explain.

Your answer:
"""

def render_user_prompt(query: str, context_str: str) -> str:
    return USER_PROMPT_TEMPLATE.format(query=query, sources_block=context_str)
