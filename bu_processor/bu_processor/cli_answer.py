#!/usr/bin/env python3
"""
CLI Answer Demo: Complete Retrieval → Answer Synthesis Pipeline

Demonstrates the full pipeline from query to cited answer using the 
hybrid retriever and answer synthesis system.

Usage:
    python cli_answer.py "What is professional liability insurance?"
    python cli_answer.py "How does corporate finance work?"
"""

import sys
from bu_processor.factories import make_hybrid_retriever, synthesize

def ask(query: str):
    """Ask a question and get a synthesized answer with citations."""
    print(f"🔍 Question: {query}")
    print("=" * 60)
    
    # Step 1: Retrieve relevant documents
    print("📄 Retrieving relevant documents...")
    retriever = make_hybrid_retriever()
    if not retriever:
        print("❌ Retriever not configured properly")
        return
    
    hits = retriever.retrieve(query, final_top_k=8)
    print(f"   Found {len(hits)} relevant documents")
    
    if not hits:
        print("❌ No relevant documents found")
        return
    
    # Step 2: Synthesize answer
    print("🤖 Synthesizing answer...")
    try:
        result = synthesize(query, hits)
    except Exception as e:
        print(f"❌ Error synthesizing answer: {e}")
        return
    
    # Step 3: Display answer
    print()
    print("📋 ANSWER:")
    print("-" * 40)
    print(result.text)
    print()
    
    # Step 4: Display sources
    if result.sources_table:
        print("📚 SOURCES:")
        print("-" * 40)
        for i, s in enumerate(result.sources_table, 1):
            chunk_id = s.get('chunk_id', 'Unknown')
            doc_id = s.get('doc_id', 'Unknown') 
            section = s.get('section', '')
            page = s.get('page', '')
            
            source_line = f"[{i}] chunk={chunk_id} doc={doc_id}"
            if section:
                source_line += f" sec={section}"
            if page:
                source_line += f" p={page}"
            
            print(source_line)
        print()
    
    # Step 5: Display citations (if any)
    if result.citations:
        print("📖 CITATIONS:")
        print("-" * 40)
        for citation in result.citations:
            print(f"  Paragraph {citation.paragraph_idx}: {citation.chunk_id} (from {citation.doc_id})")
        print()
    
    # Step 6: Display trace info
    if result.trace:
        print("🔍 ANALYSIS:")
        print("-" * 40)
        for key, value in result.trace.items():
            print(f"  {key}: {value}")

def main():
    """CLI main function."""
    if len(sys.argv) < 2:
        print("Usage: python cli_answer.py <question>")
        print()
        print("Examples:")
        print('  python cli_answer.py "What is professional liability insurance?"')
        print('  python cli_answer.py "How does corporate finance work?"') 
        print('  python cli_answer.py "What are the benefits of cloud computing?"')
        return
    
    query = " ".join(sys.argv[1:])  # Join all arguments as the query
    ask(query)

if __name__ == "__main__":
    main()
