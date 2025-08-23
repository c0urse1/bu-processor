#!/usr/bin/env python3
"""
🔍 PINECONE LABELS QUALITY CHECK
==============================

Quick check to verify that classification labels are properly stored
in Pinecone metadata after bulk ingestion.
"""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

def check_pinecone_labels():
    """Check Pinecone index for classification metadata"""
    try:
        import pinecone
        
        # Get configuration from environment
        api_key = os.environ.get("PINECONE_API_KEY") or os.environ.get("BU_VECTOR_DB__PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT", os.getenv("BU_VECTOR_DB__PINECONE_ENVIRONMENT", "us-east-1-aws"))
        index_name = os.getenv("PINECONE_INDEX", os.getenv("BU_VECTOR_DB__PINECONE_INDEX_NAME", "bu-processor-embeddings"))
        
        if not api_key:
            print("❌ PINECONE_API_KEY not found in environment")
            print("   Set PINECONE_API_KEY or BU_VECTOR_DB__PINECONE_API_KEY")
            return False
        
        print(f"🔍 Checking Pinecone index: {index_name}")
        print(f"📍 Environment: {environment}")
        print(f"🔑 API Key: {api_key[:8]}...")
        
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Get index
        idx = pinecone.Index(index_name)
        
        # Get index stats
        print(f"\n📊 Index Statistics:")
        stats = idx.describe_index_stats()
        print(f"   Total vectors: {stats.get('total_vector_count', 0):,}")
        
        if stats.get('namespaces'):
            for ns, ns_stats in stats['namespaces'].items():
                print(f"   Namespace '{ns}': {ns_stats.get('vector_count', 0):,} vectors")
        
        # Query for some vectors to check metadata
        print(f"\n🔍 Sample Query to Check Metadata:")
        try:
            # Simple query to get some vectors
            query_response = idx.query(
                vector=[0.1] * 1536,  # Dummy vector for text-embedding-3-small
                top_k=3,
                include_metadata=True
            )
            
            if query_response.matches:
                print(f"   Found {len(query_response.matches)} sample vectors")
                
                for i, match in enumerate(query_response.matches, 1):
                    print(f"\n   📄 Vector {i} (ID: {match.id[:20]}...):")
                    print(f"      Score: {match.score:.4f}")
                    
                    if match.metadata:
                        # Check for classification metadata
                        meta = match.metadata
                        
                        # File info
                        if 'file_name' in meta:
                            print(f"      File: {meta['file_name']}")
                        
                        # Classification results
                        classification_fields = ['predicted_label', 'predicted_category', 'predicted_confidence']
                        classification_found = any(field in meta for field in classification_fields)
                        
                        if classification_found:
                            print(f"      ✅ Classification metadata found:")
                            if 'predicted_label' in meta:
                                print(f"         Label: {meta['predicted_label']}")
                            if 'predicted_confidence' in meta:
                                print(f"         Confidence: {meta['predicted_confidence']}")
                            if 'predicted_category' in meta:
                                print(f"         Category: {meta['predicted_category']}")
                        else:
                            print(f"      ❌ No classification metadata found")
                        
                        # Other metadata
                        other_fields = ['source', 'page_count', 'text_length', 'ingestion_method']
                        other_meta = {k: v for k, v in meta.items() if k in other_fields}
                        if other_meta:
                            print(f"      Other metadata: {other_meta}")
                    else:
                        print(f"      ❌ No metadata found")
            else:
                print("   ❌ No vectors found in query")
                
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
        
        return True
        
    except ImportError:
        print("❌ Pinecone library not available")
        print("   Install with: pip install pinecone-client")
        return False
    except Exception as e:
        print(f"❌ Error checking Pinecone: {e}")
        return False

def main():
    """Main function"""
    print("🔍 Pinecone Labels Quality Check")
    print("=" * 40)
    
    success = check_pinecone_labels()
    
    if success:
        print(f"\n✅ Pinecone check completed")
    else:
        print(f"\n❌ Pinecone check failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
