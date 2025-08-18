# bu_processor/semantic/tokens.py
def approx_token_count(text: str) -> int:
    """
    Fast, model-agnostic approximation (~1.3 words/token for English/German).
    
    Args:
        text: Input text to count tokens for
        
    Returns:
        Approximate token count
    """
    words = len(text.split())
    return max(1, int(round(words / 1.3)))
