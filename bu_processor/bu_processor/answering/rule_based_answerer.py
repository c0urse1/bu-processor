from __future__ import annotations
from typing import List, Dict, Any
import re
from bu_processor.answering.models import AnswerResult, Citation

class RuleBasedAnswerer:
    """
    Deterministic, offline answer synthesis.
    Uses template-based generation with citation insertion.
    """
    
    def answer(self, query: str, packed_context: str, sources_table: List[Dict[str, Any]]) -> AnswerResult:
        """
        Generate answer using rule-based approach.
        """
        # Extract source sections from packed context
        source_sections = self._parse_packed_context(packed_context)
        
        # Generate answer paragraphs
        paragraphs = self._generate_paragraphs(query, source_sections)
        
        # Insert citations
        answer_text, citations = self._insert_citations(paragraphs, source_sections, sources_table)
        
        return AnswerResult(
            text=answer_text,
            citations=citations,
            sources_table=sources_table,
            trace={
                "method": "rule_based",
                "query": query,
                "source_count": len(sources_table),
                "paragraph_count": len(paragraphs)
            }
        )
    
    def _parse_packed_context(self, packed_context: str) -> Dict[int, str]:
        """Parse packed context into source sections."""
        sections = {}
        current_source = None
        current_content = []
        
        for line in packed_context.split('\n'):
            # Look for source headers like "[1] Title (doc:xyz)"
            header_match = re.match(r'\[(\d+)\]\s*(.+)', line.strip())
            if header_match:
                # Save previous section
                if current_source is not None and current_content:
                    sections[current_source] = '\n'.join(current_content).strip()
                
                # Start new section
                current_source = int(header_match.group(1))
                current_content = []
            elif current_source is not None and line.strip():
                current_content.append(line)
        
        # Save final section
        if current_source is not None and current_content:
            sections[current_source] = '\n'.join(current_content).strip()
        
        return sections
    
    def _generate_paragraphs(self, query: str, source_sections: Dict[int, str]) -> List[str]:
        """Generate answer paragraphs based on query and sources."""
        if not source_sections:
            return ["No relevant information found in the sources."]
        
        # Extract key terms from query
        query_terms = set(re.findall(r'\w+', query.lower()))
        query_terms = {term for term in query_terms if len(term) > 2}  # Filter short words
        
        paragraphs = []
        
        # Strategy 1: Direct information extraction
        if self._is_factual_query(query):
            paragraphs.extend(self._extract_factual_information(query, source_sections, query_terms))
        
        # Strategy 2: Comparison or analysis
        elif self._is_comparison_query(query):
            paragraphs.extend(self._generate_comparison(query, source_sections, query_terms))
        
        # Strategy 3: General summarization
        else:
            paragraphs.extend(self._generate_summary(query, source_sections, query_terms))
        
        return paragraphs if paragraphs else ["Based on the available sources, information related to your query was found but could not be synthesized into a coherent answer."]
    
    def _is_factual_query(self, query: str) -> bool:
        """Check if query asks for specific facts."""
        factual_indicators = ['what is', 'who is', 'when did', 'where is', 'how much', 'what does']
        return any(indicator in query.lower() for indicator in factual_indicators)
    
    def _is_comparison_query(self, query: str) -> bool:
        """Check if query asks for comparison."""
        comparison_indicators = ['difference', 'compare', 'versus', 'vs', 'better', 'worse', 'similar']
        return any(indicator in query.lower() for indicator in comparison_indicators)
    
    def _extract_factual_information(self, query: str, source_sections: Dict[int, str], query_terms: set) -> List[str]:
        """Extract direct factual information."""
        paragraphs = []
        
        # Find sentences that contain query terms
        relevant_sentences = []
        for source_id, content in source_sections.items():
            sentences = re.split(r'[.!?]+', content)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_terms = set(re.findall(r'\w+', sentence.lower()))
                if query_terms.intersection(sentence_terms):
                    relevant_sentences.append((source_id, sentence))
        
        if relevant_sentences:
            # Group by source and create paragraphs
            current_paragraph = []
            current_source = None
            
            for source_id, sentence in relevant_sentences[:3]:  # Limit to top 3 relevant sentences
                if current_source != source_id:
                    if current_paragraph:
                        paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = [sentence]
                    current_source = source_id
                else:
                    current_paragraph.append(sentence)
            
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
        
        return paragraphs
    
    def _generate_comparison(self, query: str, source_sections: Dict[int, str], query_terms: set) -> List[str]:
        """Generate comparison-based answer."""
        paragraphs = []
        
        # Simple approach: create paragraph per source showing relevant info
        for source_id, content in source_sections.items():
            sentences = re.split(r'[.!?]+', content)
            relevant_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_terms = set(re.findall(r'\w+', sentence.lower()))
                if query_terms.intersection(sentence_terms):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                paragraphs.append(' '.join(relevant_sentences[:2]))  # Max 2 sentences per source
        
        return paragraphs
    
    def _generate_summary(self, query: str, source_sections: Dict[int, str], query_terms: set) -> List[str]:
        """Generate general summary."""
        paragraphs = []
        
        # Extract first few sentences from each source that mention query terms
        for source_id, content in source_sections.items():
            sentences = re.split(r'[.!?]+', content)
            selected_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_terms = set(re.findall(r'\w+', sentence.lower()))
                if query_terms.intersection(sentence_terms) or len(selected_sentences) == 0:
                    selected_sentences.append(sentence)
                    if len(selected_sentences) >= 2:
                        break
            
            if selected_sentences:
                paragraphs.append(' '.join(selected_sentences))
        
        return paragraphs
    
    def _insert_citations(self, paragraphs: List[str], source_sections: Dict[int, str], sources_table: List[Dict[str, Any]]) -> tuple[str, List[Citation]]:
        """Insert citation markers and build citation list."""
        citations = []
        cited_paragraphs = []
        
        for para_idx, paragraph in enumerate(paragraphs):
            # Find which sources this paragraph draws from
            citing_sources = set()
            para_terms = set(re.findall(r'\w+', paragraph.lower()))
            
            for source_id, content in source_sections.items():
                content_terms = set(re.findall(r'\w+', content.lower()))
                # If paragraph shares significant terms with source content
                overlap = para_terms.intersection(content_terms)
                if len(overlap) >= 3:  # Threshold for citation
                    citing_sources.add(source_id)
            
            # Add citations to paragraph
            if citing_sources:
                citation_marks = ', '.join(f'[{src}]' for src in sorted(citing_sources))
                cited_paragraph = f"{paragraph} {citation_marks}"
                cited_paragraphs.append(cited_paragraph)
                
                # Record citations
                for source_id in citing_sources:
                    if source_id <= len(sources_table):  # Ensure valid index
                        citations.append(Citation(
                            paragraph_idx=para_idx,
                            chunk_id=sources_table[source_id - 1]['chunk_id'],
                            doc_id=sources_table[source_id - 1]['doc_id']
                        ))
            else:
                cited_paragraphs.append(paragraph)
        
        answer_text = '\n\n'.join(cited_paragraphs)
        return answer_text, citations
