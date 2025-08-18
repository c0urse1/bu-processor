#!/usr/bin/env python3
"""
Demo CLI for Answer Synthesis Pipeline
Shows complete flow: query → retrieval → reranking → summarization → answer synthesis
"""

from bu_processor.factories import (
    make_embeddings_backend, make_vector_index, make_storage,
    make_upsert_pipeline, make_hybrid_retriever, make_reranker,
    make_summarizer, make_answer_pipeline
)
from bu_processor.config import settings

def demo_answer_synthesis():
    """Demo the complete answer synthesis pipeline."""
    
    print("🔧 Setting up Answer Synthesis Pipeline...")
    print(f"Configuration: {settings.ANSWERER_BACKEND} answerer, {settings.ANSWER_TOKEN_BUDGET} token budget")
    
    # Initialize components
    embeddings = make_embeddings_backend()
    index = make_vector_index()
    storage = make_storage()
    
    # Create sample documents
    documents = [
        {
            "id": "doc1_chunk1",
            "text": "Insurance coverage protects against financial losses. Health insurance specifically covers medical expenses and hospital bills. Most policies require monthly premium payments.",
            "metadata": {"doc_id": "insurance_guide", "section": "overview", "page": 1}
        },
        {
            "id": "doc1_chunk2", 
            "text": "Claims processing typically takes 5-10 business days. Policyholders must submit documentation including medical records and receipts. Pre-authorization may be required for major procedures.",
            "metadata": {"doc_id": "insurance_guide", "section": "claims", "page": 3}
        },
        {
            "id": "doc2_chunk1",
            "text": "Corporate liability insurance protects businesses from lawsuits and damages. Professional liability covers errors and omissions in professional services. General liability covers property damage and bodily injury.",
            "metadata": {"doc_id": "business_insurance", "section": "types", "page": 1}
        },
        {
            "id": "doc2_chunk2",
            "text": "Risk assessment determines premium costs. Factors include industry type, company size, and claims history. Higher risk businesses pay significantly more for coverage.",
            "metadata": {"doc_id": "business_insurance", "section": "pricing", "page": 5}
        }
    ]
    
    # Index documents
    print("\n📚 Indexing sample documents...")
    upsert_pipeline = make_upsert_pipeline()
    for doc in documents:
        upsert_pipeline.upsert_document(
            doc_id=doc["id"],
            text=doc["text"],
            metadata=doc["metadata"]
        )
    
    # Set up retrieval and answer synthesis
    retriever = make_hybrid_retriever()
    reranker = make_reranker()
    summarizer = make_summarizer()
    answer_pipeline = make_answer_pipeline()
    
    if not answer_pipeline:
        print("❌ Answer synthesis disabled in configuration")
        return
    
    # Demo queries
    queries = [
        "What is insurance coverage and how does it work?",
        "How long does claims processing take?", 
        "What factors affect insurance premium costs?",
        "Compare health insurance vs business liability insurance"
    ]
    
    print(f"\n🎯 Testing {len(queries)} queries with complete pipeline...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        print("=" * 60)
        
        # Step 1: Retrieve
        hits = retriever.retrieve(query, k=4)
        print(f"🔍 Retrieved {len(hits)} hits")
        
        # Step 2: Rerank 
        if reranker and len(hits) > 1:
            hits = reranker.rerank(query, hits)
            print(f"📊 Reranked results")
        
        # Step 3: Summarize chunks (optional enhancement)
        if summarizer:
            for hit in hits:
                summary = summarizer.summarize(query, hit.text)
                hit.metadata["summary"] = summary
            print(f"📝 Generated summaries")
        
        # Step 4: Answer synthesis
        answer_result = answer_pipeline.synthesize_answer(query, hits)
        
        print(f"\n💬 Answer:")
        print(answer_result.text)
        
        if answer_result.citations:
            print(f"\n📚 Citations ({len(answer_result.citations)}):")
            for cite in answer_result.citations:
                print(f"  Para {cite.paragraph_idx}: {cite.chunk_id} from {cite.doc_id}")
        
        if answer_result.sources_table:
            print(f"\n📖 Sources:")
            for i, source in enumerate(answer_result.sources_table, 1):
                doc_info = f"{source['doc_id']}"
                if source.get('section'):
                    doc_info += f" ({source['section']})"
                if source.get('page'):
                    doc_info += f" p.{source['page']}"
                print(f"  [{i}] {doc_info} (score: {source.get('score', 'N/A'):.3f})")
        
        if answer_result.trace:
            print(f"\n🔧 Method: {answer_result.trace.get('method', 'unknown')}")
            if answer_result.trace.get('grounding_warnings'):
                print(f"⚠️  Warnings: {', '.join(answer_result.trace['grounding_warnings'])}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    demo_answer_synthesis()
