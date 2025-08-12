import sys
sys.path.insert(0, '.')

# Test 1: Import test
print("Test 1: Import tests")
try:
    from bu_processor.pipeline.classifier import RealMLClassifier
    print("✅ RealMLClassifier import works")
except Exception as e:
    print(f"❌ RealMLClassifier import failed: {e}")

# Test 2: SimHash constructor
print("\nTest 2: SimHash constructor")
try:
    from bu_processor.pipeline.simhash_semantic_deduplication import SemanticSimHashGenerator
    generator = SemanticSimHashGenerator(bit_size=64, ngram_size=3)
    print(f"✅ SemanticSimHashGenerator constructor works - bit_size: {generator.bit_size}")
except Exception as e:
    print(f"❌ SemanticSimHashGenerator constructor failed: {e}")

# Test 3: Calculate simhash
print("\nTest 3: Calculate simhash")
try:
    from bu_processor.pipeline.simhash_semantic_deduplication import calculate_simhash
    result = calculate_simhash("test text", bit_size=64, ngram_size=3)
    print(f"✅ calculate_simhash works - result: {result}")
except Exception as e:
    print(f"❌ calculate_simhash failed: {e}")

print("\n🏁 Quick validation complete!")
