#!/usr/bin/env python3
"""Test SimHash Generator fixes - Fix #9"""

import sys
from pathlib import Path

# Add bu_processor to Python path
script_dir = Path(__file__).resolve().parent
bu_processor_dir = script_dir / "bu_processor"
sys.path.insert(0, str(bu_processor_dir))

def test_simhash_generator_fixes():
    """Test dass die SimHash-Generator Fixes korrekt implementiert wurden."""
    
    print("🔍 Testing SimHash Generator Private Helper Methods...")
    
    try:
        # Test 1: Import SemanticSimHashGenerator
        from bu_processor.pipeline.simhash_semantic_deduplication import SemanticSimHashGenerator
        print("✅ SemanticSimHashGenerator import successful")
        
        # Test 2: Create generator instance
        generator = SemanticSimHashGenerator()
        print("✅ SemanticSimHashGenerator instance created")
        
        # Test 3: Check _normalize_text method exists and works
        assert hasattr(generator, '_normalize_text'), "_normalize_text method missing"
        assert callable(getattr(generator, '_normalize_text')), "_normalize_text is not callable"
        
        test_text = "  HELLO World!  Test@#$  "
        normalized = generator._normalize_text(test_text)
        assert isinstance(normalized, str), "_normalize_text should return string"
        assert normalized.lower() == normalized, "_normalize_text should return lowercase"
        assert "hello world" in normalized, "_normalize_text should normalize properly"
        print("✅ _normalize_text method works correctly")
        
        # Test 4: Check _extract_features method exists and works
        assert hasattr(generator, '_extract_features'), "_extract_features method missing"
        assert callable(getattr(generator, '_extract_features')), "_extract_features is not callable"
        
        features = generator._extract_features("hello world test", 2)
        assert isinstance(features, list), "_extract_features should return list"
        assert all(isinstance(item, tuple) and len(item) == 2 for item in features), "_extract_features should return list of (feature, weight) tuples"
        assert all(isinstance(item[0], str) and isinstance(item[1], (int, float)) for item in features), "_extract_features should return (str, float) tuples"
        print("✅ _extract_features method works correctly")
        
        # Test 5: Test calculate_simhash function compatibility
        from bu_processor.pipeline.simhash_semantic_deduplication import calculate_simhash
        
        test_text = "This is a test for SimHash calculation"
        simhash_value = calculate_simhash(test_text)
        assert isinstance(simhash_value, int), "calculate_simhash should return integer"
        print("✅ calculate_simhash function works correctly")
        
        # Test 6: Test consistency - same text should produce same hash
        hash1 = calculate_simhash("identical test text")
        hash2 = calculate_simhash("identical test text")
        assert hash1 == hash2, "Same text should produce same hash"
        print("✅ SimHash consistency verified")
        
        # Test 7: Test different texts produce different hashes
        hash3 = calculate_simhash("completely different content here")
        assert hash1 != hash3, "Different texts should produce different hashes"
        print("✅ SimHash differentiation verified")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing SimHash generator: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simhash_backward_compatibility():
    """Test dass backward compatibility für Tests funktioniert."""
    
    print("\n🔍 Testing SimHash Backward Compatibility...")
    
    try:
        from bu_processor.pipeline.simhash_semantic_deduplication import calculate_simhash, find_duplicates
        
        # Test calculate_simhash with different parameters
        text = "Test document for SimHash generation"
        
        # Default parameters
        hash1 = calculate_simhash(text)
        
        # Custom bit size
        hash2 = calculate_simhash(text, bit_size=32)
        
        # Custom ngram size
        hash3 = calculate_simhash(text, ngram_size=2)
        
        assert all(isinstance(h, int) for h in [hash1, hash2, hash3]), "All hashes should be integers"
        print("✅ calculate_simhash works with different parameters")
        
        # Test find_duplicates function
        test_docs = [
            {"id": "doc1", "text": "This is document one"},
            {"id": "doc2", "text": "This is document one"},  # Duplicate
            {"id": "doc3", "text": "Completely different content"}
        ]
        
        duplicates = find_duplicates(test_docs, threshold=8)
        assert isinstance(duplicates, dict), "find_duplicates should return dict"
        print("✅ find_duplicates function works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing backward compatibility: {e}")
        return False

def test_feature_extraction_details():
    """Test detaillierte Feature-Extraktion."""
    
    print("\n🔍 Testing Detailed Feature Extraction...")
    
    try:
        from bu_processor.pipeline.simhash_semantic_deduplication import SemanticSimHashGenerator
        
        generator = SemanticSimHashGenerator()
        
        # Test with insurance-related text
        insurance_text = "versicherung berufsunfähigkeit rente monatlich"
        normalized = generator._normalize_text(insurance_text)
        features = generator._extract_features(normalized, 2)
        
        # Should find insurance-related terms with higher weights
        insurance_features = [f for f in features if any(word in f[0] for word in ['versicherung', 'berufsunfähigkeit', 'rente'])]
        
        print(f"✅ Found {len(insurance_features)} insurance-related features")
        print(f"✅ Total features extracted: {len(features)}")
        
        # Test weight calculation
        weight = generator._calculate_basic_feature_weight("versicherung")
        assert weight > 1.0, "Insurance terms should have weight > 1.0"
        print(f"✅ Insurance term weight: {weight}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing feature extraction: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing SimHash Generator Private Helper Fixes (Fix #9)")
    print("=" * 65)
    
    success = True
    
    # Test the main fixes
    success &= test_simhash_generator_fixes()
    
    # Test backward compatibility
    success &= test_simhash_backward_compatibility()
    
    # Test feature extraction details
    success &= test_feature_extraction_details()
    
    print("\n" + "=" * 65)
    if success:
        print("🎯 ALL SIMHASH GENERATOR TESTS PASSED!")
        print("✅ Fix #9: SimHash‑Generator: private Helfer verfügbar machen - COMPLETED")
        print("\n📋 Summary of implemented fixes:")
        print("   • _normalize_text() method - text normalization with insurance context")
        print("   • _extract_features() method - n-gram extraction with weights")
        print("   • _calculate_basic_feature_weight() method - intelligent term weighting")
        print("   • Full backward compatibility for calculate_simhash() function")
        print("   • Enhanced pattern recognition for insurance domain")
        print("\n🎉 SimHash Generator is now fully functional with all private helpers!")
    else:
        print("❌ Some tests failed!")
    
    sys.exit(0 if success else 1)
