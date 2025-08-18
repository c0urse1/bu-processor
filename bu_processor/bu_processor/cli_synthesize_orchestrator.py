#!/usr/bin/env python3
"""
CLI Demo: Answer Synthesis with Grounding Orchestrator

Demonstrates the complete answer synthesis orchestrator that includes:
- Context packing with token budgets
- Grounding checks (confidence, conflicts)
- Citation-aware answer generation
- Deterministic and LLM-based answerers

Usage:
    python cli_synthesize_orchestrator.py "What are cloud computing benefits?"
    python cli_synthesize_orchestrator.py "Compare costs: 100k vs 200k users" --allow-conflicts
    python cli_synthesize_orchestrator.py "How does Docker work?" --backend openai_simple
    python cli_synthesize_orchestrator.py "What is machine learning?" --min-confidence 0.5
"""

import sys
from typing import List
from bu_processor.factories import make_synthesize_orchestrator
from bu_processor.retrieval.models import RetrievalHit
from bu_processor.config import settings

def create_sample_documents() -> List[RetrievalHit]:
    """Create sample documents with varying confidence levels."""
    sample_docs = [
        {
            "id": "doc1_chunk1",
            "text": "Cloud computing provides scalable, on-demand access to computing resources over the internet. Studies show cost savings of 20-30% for most enterprises moving to cloud infrastructure.",
            "metadata": {
                "doc_id": "cloud_guide_2024",
                "title": "Cloud Computing Fundamentals",
                "section": "Cost Benefits Analysis",
                "page": 1
            },
            "score": 0.85
        },
        {
            "id": "doc1_chunk2", 
            "text": "Traditional infrastructure costs vary significantly. Some reports indicate savings of 15-25% with cloud adoption, while others suggest costs may increase initially due to migration expenses.",
            "metadata": {
                "doc_id": "cloud_analysis_2023",
                "title": "Cloud Migration Study", 
                "section": "Financial Impact Assessment",
                "page": 3
            },
            "score": 0.78
        },
        {
            "id": "doc2_chunk1",
            "text": "Docker containers enable application portability across environments. Container orchestration platforms like Kubernetes manage deployment, scaling, and operations of containerized applications.",
            "metadata": {
                "doc_id": "docker_handbook",
                "title": "Container Technology Guide",
                "section": "Docker Fundamentals", 
                "page": 12
            },
            "score": 0.82
        },
        {
            "id": "doc3_chunk1",
            "text": "Machine learning algorithms learn patterns from data to make predictions. Popular frameworks include TensorFlow, PyTorch, and scikit-learn for building ML models.",
            "metadata": {
                "doc_id": "ml_introduction",
                "title": "Machine Learning Basics",
                "section": "Introduction to ML",
                "page": 5
            },
            "score": 0.79
        },
        {
            "id": "doc4_chunk1",
            "text": "Cloud pricing models vary by provider. AWS charges approximately $100 per month for small workloads, while Azure may cost $150-200 for similar configurations.",
            "metadata": {
                "doc_id": "cloud_pricing_comparison",
                "title": "Cloud Cost Analysis",
                "section": "Provider Comparison",
                "page": 8
            },
            "score": 0.73
        },
        {
            "id": "doc5_chunk1", 
            "text": "Recent studies indicate cloud costs range from $50-300 monthly for small business workloads, depending on usage patterns and service selections.",
            "metadata": {
                "doc_id": "small_business_cloud_study",
                "title": "SMB Cloud Adoption",
                "section": "Cost Analysis",
                "page": 2
            },
            "score": 0.71
        }
    ]
    
    return [RetrievalHit(**doc) for doc in sample_docs]

def simulate_retrieval_with_confidence(query: str, docs: List[RetrievalHit]) -> List[RetrievalHit]:
    """Simulate retrieval with realistic confidence scoring."""
    query_words = set(query.lower().split())
    scored_docs = []
    
    for doc in docs:
        text_words = set(doc.text.lower().split())
        title_words = set(doc.metadata.get("title", "").lower().split())
        section_words = set(doc.metadata.get("section", "").lower().split())
        
        # Calculate relevance score
        text_overlap = len(query_words.intersection(text_words))
        title_overlap = len(query_words.intersection(title_words)) * 2
        section_overlap = len(query_words.intersection(section_words)) * 1.5
        
        total_score = (text_overlap + title_overlap + section_overlap) / max(len(query_words), 1)
        
        # Add some variation to simulate real retrieval scores
        if "cloud" in query.lower() and "cloud" in doc.text.lower():
            total_score *= 1.2
        if "docker" in query.lower() and "docker" in doc.text.lower():
            total_score *= 1.3
        if "machine learning" in query.lower() and "machine learning" in doc.text.lower():
            total_score *= 1.1
        
        if total_score > 0.1:
            scored_docs.append(RetrievalHit(
                id=doc.id,
                text=doc.text,
                metadata=doc.metadata,
                score=min(1.0, total_score)  # Cap at 1.0
            ))
    
    return sorted(scored_docs, key=lambda x: x.score, reverse=True)[:5]

def demo_synthesize_orchestrator(
    query: str, 
    min_confidence: float = None,
    allow_conflicts: bool = False,
    answerer_backend: str = None,
    token_budget: int = None
):
    """Demonstrate the synthesize orchestrator with grounding checks."""
    print(f"🔍 Query: {query}")
    print("=" * 70)
    
    # Override settings if specified
    original_settings = {}
    if answerer_backend:
        original_settings['ANSWERER_BACKEND'] = settings.ANSWERER_BACKEND
        settings.ANSWERER_BACKEND = answerer_backend
        print(f"🔧 Using answerer backend: {answerer_backend}")
    
    if min_confidence is not None:
        original_settings['ANSWER_MIN_CONFIDENCE'] = settings.ANSWER_MIN_CONFIDENCE  
        settings.ANSWER_MIN_CONFIDENCE = min_confidence
        print(f"🎯 Minimum confidence: {min_confidence}")
    
    if token_budget is not None:
        original_settings['ANSWER_TOKEN_BUDGET'] = settings.ANSWER_TOKEN_BUDGET
        settings.ANSWER_TOKEN_BUDGET = token_budget
        print(f"📊 Token budget: {token_budget}")
    
    print(f"🔄 Allow conflicts: {allow_conflicts}")
    print()
    
    try:
        # Step 1: Get orchestrator
        synthesize_func = make_synthesize_orchestrator()
        if not synthesize_func:
            print("❌ Answer synthesis disabled or not configured")
            return
        
        # Step 2: Simulate retrieval
        sample_docs = create_sample_documents()
        retrieved_hits = simulate_retrieval_with_confidence(query, sample_docs)
        
        print(f"📄 Retrieved {len(retrieved_hits)} relevant documents:")
        for i, hit in enumerate(retrieved_hits, 1):
            title = hit.metadata.get('title', 'Unknown')
            score = hit.score
            print(f"  [{i}] {title} (confidence: {score:.3f})")
        print()
        
        if not retrieved_hits:
            print("❌ No relevant documents found")
            return
        
        # Step 3: Use synthesize orchestrator
        print("🤖 Running synthesize orchestrator...")
        print(f"   Grounding checks: confidence threshold = {settings.ANSWER_MIN_CONFIDENCE}")
        print(f"   Context packing: token budget = {settings.ANSWER_TOKEN_BUDGET}")
        print(f"   Conflict handling: allow = {allow_conflicts}")
        print()
        
        # Call the orchestrator
        result = synthesize_func(
            query=query,
            hits=retrieved_hits,
            allow_conflicts=allow_conflicts
        )
        
        # Step 4: Display results
        print("📋 SYNTHESIZED ANSWER:")
        print("-" * 50)
        print(result.text)
        print()
        
        # Display grounding information
        if result.trace:
            print("🔍 GROUNDING ANALYSIS:")
            print("-" * 50)
            confidence = result.trace.get('confidence', 'N/A')
            conflict = result.trace.get('conflict', False)
            grounding_passed = result.trace.get('grounding_passed', False)
            
            print(f"  Confidence Score: {confidence}")
            print(f"  Conflicts Detected: {conflict}")
            print(f"  Grounding Passed: {grounding_passed}")
            
            if 'reason' in result.trace:
                print(f"  Failure Reason: {result.trace['reason']}")
            
            if 'sources_used' in result.trace:
                print(f"  Sources Used: {result.trace['sources_used']}")
            
            if 'context_tokens' in result.trace:
                print(f"  Context Tokens: {result.trace['context_tokens']:.0f}")
            print()
        
        # Display citations
        if result.citations:
            print("📖 CITATIONS:")
            print("-" * 50)
            for citation in result.citations:
                chunk_id = citation.chunk_id
                doc_id = citation.doc_id
                para_idx = citation.paragraph_idx
                print(f"  Paragraph {para_idx}: {chunk_id} (from {doc_id})")
            print()
        
        # Display sources table
        if result.sources_table:
            print("📚 SOURCES USED:")
            print("-" * 50)
            for i, source in enumerate(result.sources_table, 1):
                title = source.get('title', 'Unknown')
                section = source.get('section', '')
                doc_id = source.get('doc_id', '')
                score = source.get('score', 'N/A')
                
                source_info = f"[{i}] {title}"
                if section:
                    source_info += f" → {section}"
                if doc_id:
                    source_info += f" [{doc_id}]"
                source_info += f" (score: {score})"
                
                print(f"  {source_info}")
            print()
    
    finally:
        # Restore original settings
        for key, value in original_settings.items():
            setattr(settings, key, value)

def main():
    """CLI main function."""
    if len(sys.argv) < 2:
        print("Usage: python cli_synthesize_orchestrator.py <query> [options]")
        print()
        print("Options:")
        print("  --min-confidence <float>   Minimum confidence threshold (default: 0.3)")
        print("  --allow-conflicts          Allow answering despite conflicts")
        print("  --backend <name>           Answerer backend (rule_based, openai_simple, etc.)")
        print("  --token-budget <int>       Token budget for context packing")
        print()
        print("Examples:")
        print('  python cli_synthesize_orchestrator.py "What are cloud benefits?"')
        print('  python cli_synthesize_orchestrator.py "Compare costs" --allow-conflicts')
        print('  python cli_synthesize_orchestrator.py "How does ML work?" --min-confidence 0.5')
        print('  python cli_synthesize_orchestrator.py "Docker guide" --backend openai_simple')
        return
    
    query = sys.argv[1]
    
    # Parse options
    min_confidence = None
    allow_conflicts = "--allow-conflicts" in sys.argv
    answerer_backend = None
    token_budget = None
    
    # Parse min-confidence
    if "--min-confidence" in sys.argv:
        try:
            idx = sys.argv.index("--min-confidence") + 1
            if idx < len(sys.argv):
                min_confidence = float(sys.argv[idx])
        except (IndexError, ValueError):
            print("❌ Invalid --min-confidence value")
            return
    
    # Parse backend
    if "--backend" in sys.argv:
        try:
            idx = sys.argv.index("--backend") + 1
            if idx < len(sys.argv):
                answerer_backend = sys.argv[idx]
        except IndexError:
            print("❌ Invalid --backend option")
            return
    
    # Parse token budget
    if "--token-budget" in sys.argv:
        try:
            idx = sys.argv.index("--token-budget") + 1
            if idx < len(sys.argv):
                token_budget = int(sys.argv[idx])
        except (IndexError, ValueError):
            print("❌ Invalid --token-budget value")
            return
    
    # Display configuration
    print("⚙️  CONFIGURATION:")
    print(f"   Answerer Backend: {answerer_backend or settings.ANSWERER_BACKEND}")
    print(f"   Min Confidence: {min_confidence or settings.ANSWER_MIN_CONFIDENCE}")
    print(f"   Token Budget: {token_budget or settings.ANSWER_TOKEN_BUDGET}")
    print(f"   Allow Conflicts: {allow_conflicts}")
    print()
    
    # Run demo
    demo_synthesize_orchestrator(
        query=query,
        min_confidence=min_confidence,
        allow_conflicts=allow_conflicts,
        answerer_backend=answerer_backend,
        token_budget=token_budget
    )

if __name__ == "__main__":
    main()
