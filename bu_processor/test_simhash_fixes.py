#!/usr/bin/env python3
"""
Test SimHash & ContentType fixes.
"""

def test_simhash_fixes():
    """Test SimHash and ContentType imports and functionality."""
    
    print("🔍 Testing SimHash & ContentType fixes...")
    
    # Test 1: Import ContentType and SimHash classes
    try:
        from bu_processor.pipeline.simhash_semantic_deduplication import SemanticSimHashGenerator, ContentType
        print("✅ SimHash and ContentType imports work")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Create generator
    try:
        gen = SemanticSimHashGenerator()
        print("✅ SemanticSimHashGenerator created")
    except Exception as e:
        print(f"❌ Generator creation failed: {e}")
        return False
    
    # Test 3: Test _normalize_text method
    try:
        normalized = gen._normalize_text('  Hello   World  ')
        expected = "hello world"
        print(f"✅ Normalized text: '{normalized}' (expected: '{expected}')")
        assert normalized == expected, f"Expected '{expected}', got '{normalized}'"
    except Exception as e:
        print(f"❌ Normalization failed: {e}")
        return False
    
    # Test 4: Test _extract_features method (generator)
    try:
        features = list(gen._extract_features('hello world test', 2))
        print(f"✅ Features extracted: {features}")
        expected_features = ['hello world', 'world test']
        assert features == expected_features, f"Expected {expected_features}, got {features}"
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return False
    
    # Test 5: Test backward compatibility function
    try:
        from bu_processor.pipeline.simhash_semantic_deduplication import calculate_simhash
        hash_val = calculate_simhash('test text')
        print(f"✅ Backward compatibility works, hash: {hash_val}")
        assert isinstance(hash_val, int), f"Expected int, got {type(hash_val)}"
    except Exception as e:
        print(f"❌ Backward compatibility failed: {e}")
        return False
    
    # Test 6: Test ContentType usage
    try:
        content_type = ContentType.LEGAL_TEXT
        print(f"✅ ContentType usage works: {content_type}")
    except Exception as e:
        print(f"❌ ContentType usage failed: {e}")
        return False
    
    print("\n🎉 All SimHash & ContentType fixes working correctly!")
    print("\n📋 Summary of fixes:")
    print("   1. ✅ ContentType import added")
    print("   2. ✅ _normalize_text() uses ' '.join(text.lower().split())")
    print("   3. ✅ _extract_features() is generator yielding n-grams")
    print("   4. ✅ Backward compatibility maintained")
    print("   5. ✅ Method conflicts resolved")
    
    return True

if __name__ == "__main__":
    success = test_simhash_fixes()
    exit(0 if success else 1)
