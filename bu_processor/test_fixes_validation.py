#!/usr/bin/env python3
"""
Validation script to test our main fixes
"""

def test_imports():
    """Test critical imports"""
    try:
        from bu_processor.pipeline.classifier import RealMLClassifier
        print("✅ RealMLClassifier import works")
    except Exception as e:
        print(f"❌ RealMLClassifier import failed: {e}")
    
    try:
        from bu_processor.pipeline.enhanced_integrated_pipeline import EnhancedIntegratedPipeline
        print("✅ EnhancedIntegratedPipeline import works")
    except Exception as e:
        print(f"❌ EnhancedIntegratedPipeline import failed: {e}")
    
    try:
        from bu_processor.pipeline.simhash_semantic_deduplication import SemanticSimHashGenerator
        print("✅ SemanticSimHashGenerator import works")
    except Exception as e:
        print(f"❌ SemanticSimHashGenerator import failed: {e}")

def test_simhash_constructor():
    """Test SimHash constructor fix"""
    try:
        from bu_processor.pipeline.simhash_semantic_deduplication import SemanticSimHashGenerator
        generator = SemanticSimHashGenerator(bit_size=64, ngram_size=3)
        print(f"✅ SemanticSimHashGenerator constructor works - bit_size: {generator.bit_size}")
    except Exception as e:
        print(f"❌ SemanticSimHashGenerator constructor failed: {e}")

def test_calculate_simhash():
    """Test calculate_simhash function"""
    try:
        from bu_processor.pipeline.simhash_semantic_deduplication import calculate_simhash
        result = calculate_simhash("test text", bit_size=64, ngram_size=3)
        print(f"✅ calculate_simhash works - result: {result}")
    except Exception as e:
        print(f"❌ calculate_simhash failed: {e}")

if __name__ == "__main__":
    print("🔧 Testing fixes...")
    test_imports()
    test_simhash_constructor()
    test_calculate_simhash()
    print("🏁 Validation complete!")
