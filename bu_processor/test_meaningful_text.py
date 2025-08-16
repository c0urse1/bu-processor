#!/usr/bin/env python3
"""Quick test script to verify the _is_meaningful_text function works correctly."""

import sys
import os

# Add the bu_processor directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_meaningful_text():
    """Test cases for the _is_meaningful_text function."""
    
    try:
        from bu_processor.pipeline.pdf_extractor import _is_meaningful_text, MIN_MEANINGFUL_CHARS
        
        # Should return True - meaningful text
        assert _is_meaningful_text("This is a meaningful document with proper content.") == True
        assert _is_meaningful_text("ABC123DEF456") == True  # 12 alphanumeric chars
        assert _is_meaningful_text("Hello123456") == True   # 11 alphanumeric chars
        assert _is_meaningful_text("Test123456") == True    # 10 alphanumeric chars (exactly at threshold)
        
        # Should return False - not meaningful
        assert _is_meaningful_text("") == False                              # Empty string
        assert _is_meaningful_text("   ") == False                           # Only whitespace
        assert _is_meaningful_text("!!!@@@###") == False                     # Only punctuation
        assert _is_meaningful_text("   !!!   ") == False                     # Whitespace + punctuation
        assert _is_meaningful_text("Test123") == False                       # 7 alphanumeric chars (below threshold)
        assert _is_meaningful_text("A1B2C3D4E") == False                     # 9 alphanumeric chars (below threshold)
        assert _is_meaningful_text("      A1B2C3D4E      ") == False         # 9 alphanumeric chars with spaces
        
        # Edge cases
        assert _is_meaningful_text("A" * MIN_MEANINGFUL_CHARS) == True       # Exactly at threshold
        assert _is_meaningful_text("A" * (MIN_MEANINGFUL_CHARS - 1)) == False  # Just below threshold
        
        print("✅ All _is_meaningful_text tests passed!")
        print(f"✅ MIN_MEANINGFUL_CHARS = {MIN_MEANINGFUL_CHARS}")
        
        # Test None case separately to avoid assert with None
        try:
            result = _is_meaningful_text(None)
            assert result == False
            print("✅ None test passed!")
        except Exception as e:
            print(f"❌ None test failed: {e}")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure you're running this from the correct directory")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_meaningful_text()
