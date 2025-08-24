#!/usr/bin/env python3
"""
Demo: Complete Query Path - Text and Vector Search

This script demonstrates how the query path is now complete, supporting
both direct vector queries and text-based queries with automatic embedding.
"""
import os
import sys
from pathlib import Path

# Add the bu_processor package to path
sys.path.insert(0, str(Path(__file__).parent / "bu_processor"))

def demo_query_path_completeness():
    """Demonstrate complete query path with text and vector support."""
    print("=" * 60)
    print("COMPLETE QUERY PATH DEMONSTRATION")
    print("=" * 60)
    
    print("\n🎯 Query Path Analysis:")
    print("Before: search_similar_documents required pre-computed vectors")
    print("After: query_by_text allows direct text → results")
    print("Result: Complete query path for MVP users")
    
    print("\n" + "=" * 60)
    print("AVAILABLE QUERY METHODS")
    print("=" * 60)
    
    try:
        from bu_processor.integrations.pinecone_manager import PineconeManager
        from bu_processor.embeddings.embedder import Embedder
        
        # Create instances for demonstration (won't actually connect)
        print("\n1. Creating manager and embedder instances...")
        manager = PineconeManager(
            index_name="demo-index",
            api_key="demo-key",
            force_simple=True  # Use simple implementation for demo
        )
        
        embedder = Embedder()
        print("   ✓ Manager and embedder created")
        
        # Show available query methods
        print(f"\n2. Available query methods in {manager.implementation_type}:")
        
        query_methods = [method for method in dir(manager) if 'query' in method.lower()]
        for method in query_methods:
            if not method.startswith('_'):
                print(f"   - {method}")
        
        print("\n3. Query Path Options:")
        
        # Option 1: Direct vector query
        print("\n   📝 Option 1: Direct Vector Query")
        print("   Usage: manager.query_by_vector(vector, top_k=5)")
        print("   Best for: When you already have embeddings")
        print("   Example:")
        print("     vector = [0.1, 0.2, 0.3, ...]  # Pre-computed embedding")
        print("     results = manager.query_by_vector(vector, top_k=5)")
        
        # Option 2: Text-based query (NEW - closes the gap)
        print("\n   🔍 Option 2: Text Query (MVP-Friendly)")
        print("   Usage: manager.query_by_text(text, embedder, top_k=5)")
        print("   Best for: Direct text search (MVP use case)")
        print("   Example:")
        print("     text = 'Find documents about machine learning'")
        print("     results = manager.query_by_text(text, embedder, top_k=5)")
        
        # Option 3: Legacy compatibility
        print("\n   🔧 Option 3: Legacy Compatibility")
        print("   Usage: manager.search_similar_documents(vector, top_k=5)")
        print("   Best for: Existing code migration")
        print("   Example:")
        print("     results = manager.search_similar_documents(vector, top_k=5)")
        
        print("\n4. Query Path Flow:")
        print("   Text Input → query_by_text() → embedder.encode_one() → query_by_vector() → Results")
        print("   Vector Input → query_by_vector() → Results")
        print("   Legacy Input → search_similar_documents() → Results")
        
    except ImportError as e:
        print(f"   Import error: {e}")
        print("   (Demo can still show the concept)")
    
    print("\n" + "=" * 60)
    print("MVP USER SCENARIOS")
    print("=" * 60)
    
    print("\n🚀 Scenario 1: New MVP User")
    print("   Problem: 'I have text, I want similar documents'")
    print("   Solution: manager.query_by_text(text, embedder)")
    print("   Benefit: No embedding knowledge required")
    
    print("\n🔧 Scenario 2: Performance-Conscious User")
    print("   Problem: 'I want to reuse embeddings for multiple queries'")
    print("   Solution: vector = embedder.encode_one(text)")
    print("            manager.query_by_vector(vector)")
    print("   Benefit: Embedding computed once, queried multiple times")
    
    print("\n🔄 Scenario 3: Migration User")
    print("   Problem: 'I have existing code using search_similar_documents'")
    print("   Solution: manager.search_similar_documents(vector)")
    print("   Benefit: No code changes required")
    
    print("\n" + "=" * 60)
    print("IMPLEMENTATION DETAILS")
    print("=" * 60)
    
    print("\n📋 query_by_text Implementation:")
    print("""
def query_by_text(self, text: str, embedder, top_k=5, **kwargs):
    '''Search for similar texts via embedding.'''
    vec = embedder.encode_one(text)
    return self.query_by_vector(vec, top_k=top_k, **kwargs)
""")
    
    print("🎯 Key Benefits:")
    print("   ✅ Simple one-liner implementation")
    print("   ✅ Reuses existing query_by_vector logic")
    print("   ✅ Consistent parameter handling")
    print("   ✅ Works with both simple and enhanced implementations")
    print("   ✅ Closes the 'text → results' gap for MVP users")


def demo_query_examples():
    """Show practical query examples."""
    print("\n" + "=" * 60)
    print("PRACTICAL QUERY EXAMPLES")
    print("=" * 60)
    
    print("\n📝 Example 1: Simple Text Search")
    print("""
from bu_processor.integrations.pinecone_manager import PineconeManager
from bu_processor.embeddings.embedder import Embedder

manager = PineconeManager(index_name="documents")
embedder = Embedder()

# Direct text search - MVP friendly!
results = manager.query_by_text(
    text="machine learning algorithms",
    embedder=embedder,
    top_k=5
)

for match in results.get('matches', []):
    print(f"Score: {match['score']}, Doc: {match['id']}")
""")
    
    print("\n🔍 Example 2: Advanced Text Search with Filters")
    print("""
# Text search with metadata filtering
results = manager.query_by_text(
    text="data science tutorials",
    embedder=embedder,
    top_k=10,
    include_metadata=True,
    namespace="tutorials",
    filter={"category": {"$eq": "data-science"}}
)
""")
    
    print("\n⚡ Example 3: Performance-Optimized Search")
    print("""
# Pre-compute embedding for multiple queries
query_text = "neural networks deep learning"
query_vector = embedder.encode_one(query_text)

# Use vector for multiple searches
results1 = manager.query_by_vector(query_vector, top_k=5)
results2 = manager.query_by_vector(query_vector, top_k=10, 
                                  filter={"type": "research"})
# Embedding computed only once!
""")
    
    print("\n🔄 Example 4: Legacy Code Migration")
    print("""
# Existing code using old method
def legacy_search(text, manager, embedder):
    vector = embedder.encode_one(text)
    return manager.search_similar_documents(vector, top_k=5)

# New simplified code
def modern_search(text, manager, embedder):
    return manager.query_by_text(text, embedder, top_k=5)

# Both work identically!
""")


def demo_query_path_benefits():
    """Demonstrate the benefits of complete query path."""
    print("\n" + "=" * 60)
    print("QUERY PATH BENEFITS")
    print("=" * 60)
    
    print("\n✅ COMPLETE MVP SUPPORT:")
    print("   - Text input → Direct results")
    print("   - No embedding knowledge required")
    print("   - One-line query for common use cases")
    
    print("\n✅ PERFORMANCE FLEXIBILITY:")
    print("   - Option to pre-compute embeddings")
    print("   - Reuse vectors for multiple queries")
    print("   - Batch embedding for efficiency")
    
    print("\n✅ BACKWARD COMPATIBILITY:")
    print("   - Legacy search_similar_documents preserved")
    print("   - Existing code continues to work")
    print("   - Smooth migration path")
    
    print("\n✅ CONSISTENT API:")
    print("   - Same parameter structure across methods")
    print("   - Unified namespace/filter handling")
    print("   - Predictable return formats")
    
    print("\n✅ IMPLEMENTATION SIMPLICITY:")
    print("   - query_by_text wraps query_by_vector")
    print("   - No code duplication")
    print("   - Maintains single source of truth")
    
    print("\n🎯 RESULT: No more 'ohne Vektor kein Ergebnis' problem!")
    print("   Users can now go directly from text to results")
    print("   MVP workflows fully supported")
    print("   Advanced users still have vector control")


if __name__ == "__main__":
    print("🚀 Complete Query Path Demo")
    print("=" * 60)
    print("Demonstrating text → results workflow")
    
    demo_query_path_completeness()
    demo_query_examples()
    demo_query_path_benefits()
    
    print("\n" + "=" * 60)
    print("✅ Query path gap closed!")
    print("\nBefore: Users needed to understand vectors")
    print("After:  Users can search directly with text")
    print("\nMVP-friendly query methods now available:")
    print("  - query_by_text(text, embedder)")
    print("  - query_by_vector(vector)")
    print("  - search_similar_documents(vector)  # legacy")
    print("=" * 60)
