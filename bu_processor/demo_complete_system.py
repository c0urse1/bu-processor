#!/usr/bin/env python3
"""
Demo script for the complete modular embeddings, vector search, reranking, and summarization system.

This demonstrates the full pipeline:
1. Embedding documents with SBERT/OpenAI/Fake backends
2. Indexing with FAISS or Pinecone
3. Dense + BM25 hybrid retrieval with RRF fusion
4. MMR diversification
5. Cross-encoder reranking (or heuristic for testing)
6. Query-aware extractive summarization

Usage:
    python demo_complete_system.py
    
Environment variables:
    EMBEDDINGS_BACKEND=sbert|openai|fake (default: fake for demo)
    VECTOR_INDEX=faiss|pinecone (default: faiss)
    RERANKER_BACKEND=cross_encoder|heuristic|none (default: heuristic)
    SUMMARIZER_BACKEND=extractive|none (default: extractive)
"""

import os
import tempfile
from pathlib import Path

# Set demo-friendly defaults (no network required)
os.environ.setdefault("EMBEDDINGS_BACKEND", "fake")
os.environ.setdefault("VECTOR_INDEX", "faiss") 
os.environ.setdefault("RERANKER_BACKEND", "heuristic")
os.environ.setdefault("SUMMARIZER_BACKEND", "extractive")

from bu_processor.factories import (
    make_embedder, 
    make_index, 
    make_store, 
    make_hybrid_retriever,
    make_reranker,
    make_summarizer
)
from bu_processor.pipeline.upsert_pipeline import embed_and_index_chunks
from bu_processor.config import settings

def create_sample_corpus():
    """Create sample documents for the demo"""
    return [
        {
            "doc_title": "Insurance Guide", 
            "doc_source": "internal_docs",
            "chunks": [
                {
                    "text": "Professional liability insurance protects businesses from financial losses due to claims of negligence, errors, or omissions in professional services. This coverage is essential for consultants, lawyers, doctors, and other professionals.",
                    "page": 1,
                    "section": "Professional Coverage",
                    "meta": {"doc_type": "guide", "priority": "high"}
                },
                {
                    "text": "General liability insurance covers bodily injury and property damage claims against your business. It also provides protection against advertising injury and personal injury claims.",
                    "page": 1, 
                    "section": "General Coverage",
                    "meta": {"doc_type": "guide", "priority": "medium"}
                },
                {
                    "text": "Pet insurance helps cover veterinary expenses for cats, dogs, and other pets. Most policies cover accidents, illnesses, and some preventive care depending on the plan level.",
                    "page": 2,
                    "section": "Pet Insurance", 
                    "meta": {"doc_type": "guide", "priority": "low"}
                }
            ]
        },
        {
            "doc_title": "Financial Planning Manual",
            "doc_source": "financial_docs", 
            "chunks": [
                {
                    "text": "Corporate finance involves making strategic decisions about capital structure, funding, and investment to maximize shareholder value. Key areas include debt vs equity financing, working capital management, and merger & acquisition analysis.",
                    "page": 1,
                    "section": "Corporate Finance",
                    "meta": {"doc_type": "manual", "priority": "high"}
                },
                {
                    "text": "Personal financial planning includes budgeting, saving, investing, and risk management. A comprehensive plan addresses short-term needs and long-term goals like retirement and estate planning.",
                    "page": 2,
                    "section": "Personal Finance", 
                    "meta": {"doc_type": "manual", "priority": "medium"}
                },
                {
                    "text": "Investment strategies vary based on risk tolerance, time horizon, and financial goals. Common approaches include value investing, growth investing, index fund investing, and dollar-cost averaging.",
                    "page": 3,
                    "section": "Investments",
                    "meta": {"doc_type": "manual", "priority": "medium"}
                }
            ]
        },
        {
            "doc_title": "Pet Care Handbook", 
            "doc_source": "lifestyle_docs",
            "chunks": [
                {
                    "text": "Cats are independent domestic animals that require regular veterinary care, proper nutrition, and mental stimulation. They typically sleep 12-16 hours per day and are most active during dawn and dusk.",
                    "page": 1,
                    "section": "Cat Care",
                    "meta": {"doc_type": "handbook", "priority": "low"}
                },
                {
                    "text": "Dog training requires consistency, positive reinforcement, and patience. Basic commands like sit, stay, and come should be taught early. Socialization with other dogs and people is crucial for behavioral development.",
                    "page": 2, 
                    "section": "Dog Care",
                    "meta": {"doc_type": "handbook", "priority": "low"}
                }
            ]
        }
    ]

def demonstrate_system():
    """Run complete system demonstration"""
    print("🚀 Complete Modular Embeddings & Vector Search System Demo")
    print("=" * 60)
    
    # Display configuration
    print(f"📋 Configuration:")
    print(f"   Embeddings Backend: {settings.EMBEDDINGS_BACKEND}")
    print(f"   Vector Index: {settings.VECTOR_INDEX}")
    print(f"   Reranker Backend: {settings.RERANKER_BACKEND}")
    print(f"   Summarizer Backend: {settings.SUMMARIZER_BACKEND}")
    print()
    
    # Setup temporary SQLite database
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "demo.db"
        os.environ["SQLITE_URL"] = f"sqlite:///{db_path}"
        
        print("🏗️  Initializing components...")
        # Initialize all components via factories
        embedder = make_embedder()
        index = make_index()
        store = make_store()
        
        print(f"   ✅ Embedder: {type(embedder).__name__}")
        print(f"   ✅ Index: {type(index).__name__}")
        print(f"   ✅ Store: {type(store).__name__}")
        print()
        
        print("📚 Ingesting sample documents...")
        # Ingest sample corpus
        corpus = create_sample_corpus()
        total_chunks = 0
        
        for doc in corpus:
            chunk_ids = embed_and_index_chunks(
                doc_title=doc["doc_title"],
                doc_source=doc["doc_source"], 
                doc_meta=None,
                chunks=doc["chunks"],
                embedder=embedder,
                index=index,
                store=store,
                namespace=None
            )
            total_chunks += len(chunk_ids)
            print(f"   ✅ {doc['doc_title']}: {len(chunk_ids)} chunks indexed")
        
        print(f"   📊 Total: {total_chunks} chunks in vector database")
        print()
        
        print("🔧 Building retrieval system...")
        # Now build the complete retrieval system after documents are indexed
        hybrid_retriever = make_hybrid_retriever()
        reranker = make_reranker()
        summarizer = make_summarizer()
        
        print(f"   ✅ Reranker: {type(reranker).__name__ if reranker else 'None'}")
        print(f"   ✅ Summarizer: {type(summarizer).__name__ if summarizer else 'None'}")
        print()
        
        # Demonstrate different query scenarios
        queries = [
            {
                "query": "What insurance covers professional mistakes and errors?",
                "description": "Insurance-focused query (should prioritize professional liability)",
                "expected_topics": ["professional", "liability", "insurance"]
            },
            {
                "query": "How do companies optimize their capital structure and funding?",
                "description": "Finance-focused query (should prioritize corporate finance)",
                "expected_topics": ["corporate", "finance", "capital"]
            },
            {
                "query": "Pet care insurance and veterinary coverage",
                "description": "Mixed query (should retrieve both pet care and insurance)",
                "expected_topics": ["pet", "insurance", "veterinary"]
            }
        ]
        
        print("🔍 Running retrieval demonstrations...")
        print()
        
        for i, q in enumerate(queries, 1):
            print(f"Query {i}: {q['description']}")
            print(f"📝 '{q['query']}'")
            print("-" * 50)
            
            # Perform hybrid retrieval with reranking and summarization
            hits = hybrid_retriever.retrieve(
                query=q["query"],
                final_top_k=3,
                top_k_dense=6,
                top_k_bm25=6
            )
            
            print(f"🎯 Retrieved {len(hits)} results:")
            
            for j, hit in enumerate(hits, 1):
                print(f"\n  {j}. Score: {hit.score:.3f}")
                print(f"     Text: {hit.text[:100]}...")
                print(f"     Section: {hit.metadata.get('section', 'N/A')}")
                print(f"     Doc Type: {hit.metadata.get('doc_type', 'N/A')}")
                
                # Show reranking score if available
                if "ce_score" in hit.metadata:
                    print(f"     Rerank Score: {hit.metadata['ce_score']:.3f}")
                
                # Show query-aware summary if available
                if "summary" in hit.metadata:
                    print(f"     📄 Query-aware Summary: {hit.metadata['summary']}")
            
            print("\n" + "=" * 60 + "\n")
        
        print("🎉 Demo completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("   ✅ Modular embeddings backends (SBERT/OpenAI/Fake)")
        print("   ✅ Vector indexing (FAISS/Pinecone)")
        print("   ✅ Hybrid Dense + BM25 retrieval")
        print("   ✅ RRF fusion algorithm")
        print("   ✅ MMR diversification") 
        print("   ✅ Cross-encoder reranking (or heuristic)")
        print("   ✅ Query-aware extractive summarization")
        print("   ✅ Metadata filtering support")
        print("   ✅ SQLite storage with proper cleanup")
        print()
        print("🔧 To test with production models:")
        print("   export EMBEDDINGS_BACKEND=sbert")
        print("   export RERANKER_BACKEND=cross_encoder")
        print("   export RUN_REAL_RERANKER=1")
        print("   python demo_complete_system.py")
        
        # Cleanup
        store.close()

if __name__ == "__main__":
    demonstrate_system()
