#!/usr/bin/env python3
"""
CLI Demo: Complete Answer Synthesis Pipeline

Demonstrates the full retrieval → reranking → summarization → answer synthesis pipeline
with citation tracking and grounding checks.

Usage:
    python cli_answer_synthesis.py "What are the main benefits of cloud computing?"
    python cli_answer_synthesis.py "Compare traditional vs cloud architectures" --force-answer
    python cli_answer_synthesis.py "How does Docker work?" --backend openai
"""

import sys
from typing import List
from bu_processor.factories import (
    make_dense_retriever, make_bm25_index, 
    make_reranker, make_summarizer, make_answer_pipeline
)
from bu_processor.retrieval.hybrid import HybridRetriever
from bu_processor.retrieval.models import RetrievalHit
from bu_processor.config import settings

def create_sample_documents() -> List[RetrievalHit]:
    """Create sample documents for demonstration."""
    sample_docs = [
        {
            "id": "doc1_chunk1",
            "text": "Cloud computing provides scalable, on-demand access to computing resources over the internet. Key benefits include cost reduction through pay-as-you-use models, improved scalability for handling variable workloads, and enhanced flexibility for businesses to adapt quickly to changing requirements.",
            "metadata": {
                "doc_id": "cloud_guide_2024",
                "title": "Cloud Computing Fundamentals",
                "section": "Introduction to Cloud Benefits",
                "page": 1
            },
            "score": 0.85
        },
        {
            "id": "doc1_chunk2", 
            "text": "Traditional on-premises infrastructure requires significant upfront capital investment in hardware, software licenses, and data center facilities. Organizations must also maintain dedicated IT staff for system administration, security updates, and hardware maintenance.",
            "metadata": {
                "doc_id": "cloud_guide_2024",
                "title": "Cloud Computing Fundamentals", 
                "section": "Traditional Infrastructure Challenges",
                "page": 3
            },
            "score": 0.78
        },
        {
            "id": "doc2_chunk1",
            "text": "Docker containerization technology enables applications to run consistently across different computing environments. Containers package applications with all dependencies, libraries, and configuration files needed for execution, ensuring portability between development, testing, and production systems.",
            "metadata": {
                "doc_id": "docker_handbook",
                "title": "Docker Technology Guide",
                "section": "Container Fundamentals", 
                "page": 12
            },
            "score": 0.82
        },
        {
            "id": "doc2_chunk2",
            "text": "Cloud platforms like AWS, Azure, and Google Cloud offer managed services that reduce operational overhead. These services include automated backups, security patching, monitoring, and scaling capabilities. Organizations can focus on application development rather than infrastructure management.",
            "metadata": {
                "doc_id": "cloud_platforms_comparison",
                "title": "Major Cloud Platforms Analysis",
                "section": "Managed Services Overview",
                "page": 5
            },
            "score": 0.79
        },
        {
            "id": "doc3_chunk1",
            "text": "Security in cloud environments follows a shared responsibility model. Cloud providers secure the underlying infrastructure, while customers are responsible for securing their applications, data, and access controls. This model requires clear understanding of security boundaries.",
            "metadata": {
                "doc_id": "cloud_security_best_practices",
                "title": "Cloud Security Framework",
                "section": "Shared Responsibility Model",
                "page": 8
            },
            "score": 0.73
        }
    ]
    
    return [RetrievalHit(**doc) for doc in sample_docs]

def simulate_retrieval(query: str, docs: List[RetrievalHit]) -> List[RetrievalHit]:
    """Simple simulation of retrieval based on keyword matching."""
    query_words = set(query.lower().split())
    scored_docs = []
    
    for doc in docs:
        text_words = set(doc.text.lower().split())
        title_words = set(doc.metadata.get("title", "").lower().split())
        section_words = set(doc.metadata.get("section", "").lower().split())
        
        # Calculate relevance score based on word overlap
        text_overlap = len(query_words.intersection(text_words))
        title_overlap = len(query_words.intersection(title_words)) * 2  # Title matches count more
        section_overlap = len(query_words.intersection(section_words)) * 1.5
        
        total_score = (text_overlap + title_overlap + section_overlap) / len(query_words)
        
        if total_score > 0.1:  # Only include somewhat relevant docs
            # Create new hit with updated score
            scored_docs.append(RetrievalHit(
                id=doc.id,
                text=doc.text,
                metadata=doc.metadata,
                score=total_score
            ))
    
    # Sort by score and return top results
    return sorted(scored_docs, key=lambda x: x.score, reverse=True)[:4]

def demo_answer_synthesis(query: str, force_answer: bool = False, answerer_backend: str = None):
    """Demonstrate complete answer synthesis pipeline."""
    print(f"🔍 Query: {query}")
    print("=" * 60)
    
    # Override answerer backend if specified
    if answerer_backend:
        original_backend = settings.ANSWERER_BACKEND
        settings.ANSWERER_BACKEND = answerer_backend
        print(f"🔧 Using answerer backend: {answerer_backend}")
    
    try:
        # Step 1: Simulate retrieval
        sample_docs = create_sample_documents()
        retrieved_hits = simulate_retrieval(query, sample_docs)
        
        print(f"📄 Retrieved {len(retrieved_hits)} relevant documents")
        for i, hit in enumerate(retrieved_hits, 1):
            print(f"  [{i}] {hit.metadata.get('title', 'Unknown')} (score: {hit.score:.3f})")
        
        if not retrieved_hits:
            print("❌ No relevant documents found")
            return
        
        print()
        
        # Step 2: Reranking (optional)
        reranker = make_reranker()
        if reranker:
            print("🔄 Applying reranking...")
            reranked_hits = reranker.rerank(query, retrieved_hits, top_k=4)
            print(f"   Reranked {len(reranked_hits)} results")
        else:
            reranked_hits = retrieved_hits
            print("⏭️  Skipping reranking (disabled)")
        
        # Step 3: Summarization (optional)
        summarizer = make_summarizer()
        if summarizer:
            print("📝 Applying summarization...")
            for hit in reranked_hits:
                summary = summarizer.summarize(query, [hit])
                if summary and summary.strip():
                    hit.metadata["summary"] = summary.strip()
                    print(f"   Summarized: {hit.metadata.get('title', hit.id)}")
        else:
            print("⏭️  Skipping summarization (disabled)")
        
        print()
        
        # Step 4: Answer Synthesis
        answer_pipeline = make_answer_pipeline()
        if not answer_pipeline:
            print("❌ Answer synthesis disabled")
            return
            
        print("🤖 Synthesizing answer...")
        print(f"   Force answer: {force_answer}")
        print(f"   Token budget: {settings.ANSWER_TOKEN_BUDGET}")
        print(f"   Min confidence: {settings.ANSWER_MIN_CONFIDENCE}")
        print(f"   Min sources: {settings.ANSWER_MIN_SOURCES}")
        
        # Generate answer
        result = answer_pipeline.synthesize_answer(
            query=query,
            hits=reranked_hits,
            force_answer=force_answer
        )
        
        print()
        print("📋 ANSWER:")
        print("-" * 40)
        print(result.text)
        print()
        
        # Display citations
        if result.citations:
            print("📖 CITATIONS:")
            print("-" * 40)
            for citation in result.citations:
                print(f"  Paragraph {citation.paragraph_idx}: {citation.chunk_id} (from {citation.doc_id})")
            print()
        
        # Display sources table
        if result.sources_table:
            print("📚 SOURCES:")
            print("-" * 40)
            for i, source in enumerate(result.sources_table, 1):
                title = source.get('title', 'Unknown')
                section = source.get('section', '')
                page = source.get('page', '')
                doc_id = source.get('doc_id', '')
                
                source_info = f"[{i}] {title}"
                if section:
                    source_info += f" → {section}"
                if page:
                    source_info += f" (page {page})"
                if doc_id:
                    source_info += f" [{doc_id}]"
                
                print(f"  {source_info}")
            print()
        
        # Display trace information
        if result.trace:
            print("🔍 TRACE INFO:")
            print("-" * 40)
            for key, value in result.trace.items():
                print(f"  {key}: {value}")
            print()
    
    finally:
        # Restore original backend
        if answerer_backend:
            settings.ANSWERER_BACKEND = original_backend

def main():
    """CLI main function."""
    if len(sys.argv) < 2:
        print("Usage: python cli_answer_synthesis.py <query> [--force-answer] [--backend <name>]")
        print()
        print("Examples:")
        print('  python cli_answer_synthesis.py "What are cloud computing benefits?"')
        print('  python cli_answer_synthesis.py "Compare Docker vs VMs" --force-answer')
        print('  python cli_answer_synthesis.py "How does security work?" --backend openai')
        print()
        print("Available backends: rule_based, openai")
        return
    
    query = sys.argv[1]
    force_answer = "--force-answer" in sys.argv
    
    # Parse backend option
    answerer_backend = None
    if "--backend" in sys.argv:
        try:
            backend_idx = sys.argv.index("--backend") + 1
            if backend_idx < len(sys.argv):
                answerer_backend = sys.argv[backend_idx]
        except (IndexError, ValueError):
            print("❌ Invalid --backend option")
            return
    
    # Display current configuration
    print("⚙️  CONFIGURATION:")
    print(f"   Embeddings: {settings.EMBEDDINGS_BACKEND}")
    print(f"   Vector Index: {settings.VECTOR_INDEX}")
    print(f"   Reranker: {settings.RERANKER_BACKEND}")
    print(f"   Summarizer: {settings.SUMMARIZER_BACKEND}")
    print(f"   Answerer: {answerer_backend or settings.ANSWERER_BACKEND}")
    print()
    
    # Run demo
    demo_answer_synthesis(query, force_answer, answerer_backend)

if __name__ == "__main__":
    main()
