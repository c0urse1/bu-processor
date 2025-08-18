"""
Simple test to check sentence-transformers availability
"""
try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers is available")
except ImportError as e:
    print(f"❌ sentence-transformers not available: {e}")

# Test the sbert_backend module
try:
    from bu_processor.embeddings.sbert_backend import SbertEmbeddings
    print("✅ SbertEmbeddings import successful")
except ImportError as e:
    print(f"❌ SbertEmbeddings import failed: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")

# Check what's in the module
import bu_processor.embeddings.sbert_backend as sbert_mod
print(f"Module contents: {[x for x in dir(sbert_mod) if not x.startswith('_')]}")
