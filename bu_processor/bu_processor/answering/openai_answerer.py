from __future__ import annotations
from typing import List, Dict, Any
import re
from bu_processor.answering.models import AnswerResult, Citation

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

class OpenAIAnswerer:
    """
    LLM-based answer synthesis using OpenAI GPT models.
    Generates coherent answers with proper citations.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = None, max_tokens: int = 500):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        self.model = model
        self.max_tokens = max_tokens
        
        if api_key:
            openai.api_key = api_key
    
    def answer(self, query: str, packed_context: str, sources_table: List[Dict[str, Any]]) -> AnswerResult:
        """
        Generate answer using OpenAI LLM.
        """
        # Create prompt
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(query, packed_context)
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.1,  # Low temperature for consistency
                top_p=0.9
            )
            
            answer_text = response.choices[0].message.content.strip()
            
            # Extract citations from the generated text
            citations = self._extract_citations(answer_text, sources_table)
            
            return AnswerResult(
                text=answer_text,
                citations=citations,
                sources_table=sources_table,
                trace={
                    "method": "openai",
                    "model": self.model,
                    "query": query,
                    "source_count": len(sources_table),
                    "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
                }
            )
            
        except Exception as e:
            # Fallback to error message
            return AnswerResult(
                text=f"I apologize, but I encountered an error while generating the answer: {str(e)}",
                citations=[],
                sources_table=sources_table,
                trace={
                    "method": "openai",
                    "error": str(e),
                    "query": query
                }
            )
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for answer generation."""
        return """You are a helpful AI assistant that provides accurate answers based on provided source material. Your task is to:

1. Read the user's question and the provided source material carefully
2. Generate a comprehensive answer based ONLY on the information in the sources
3. Use citation markers [1], [2], [3], etc. to reference specific sources throughout your answer
4. Structure your answer in clear paragraphs
5. If the sources don't contain enough information to answer the question, say so explicitly
6. Never make up information that isn't in the sources

Citation Guidelines:
- Place citation markers [X] immediately after statements that come from source X
- Use multiple citations [1, 2] when information comes from multiple sources
- Ensure every factual claim is properly cited
- At least one citation should appear in each paragraph

Answer Style:
- Be concise but comprehensive
- Use clear, professional language
- Structure information logically
- Focus on directly answering the user's question"""
    
    def _create_user_prompt(self, query: str, packed_context: str) -> str:
        """Create user prompt with query and context."""
        return f"""Question: {query}

Source Material:
{packed_context}

Please provide a comprehensive answer to the question based on the source material above. Remember to cite your sources using [1], [2], [3], etc. markers."""
    
    def _extract_citations(self, answer_text: str, sources_table: List[Dict[str, Any]]) -> List[Citation]:
        """Extract citation information from generated text."""
        citations = []
        
        # Find all citation markers in the text
        citation_pattern = r'\[(\d+)\]'
        paragraphs = answer_text.split('\n\n')
        
        for para_idx, paragraph in enumerate(paragraphs):
            cited_sources = re.findall(citation_pattern, paragraph)
            
            for source_str in cited_sources:
                try:
                    source_idx = int(source_str) - 1  # Convert to 0-based index
                    if 0 <= source_idx < len(sources_table):
                        citations.append(Citation(
                            paragraph_idx=para_idx,
                            chunk_id=sources_table[source_idx]['chunk_id'],
                            doc_id=sources_table[source_idx]['doc_id']
                        ))
                except (ValueError, KeyError):
                    continue  # Skip invalid citations
        
        return citations
