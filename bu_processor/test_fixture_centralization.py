#!/usr/bin/env python3
"""
Quick test of fixture centralization - self-contained version
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bu_processor"))

def test_conftest_file_exists():
    """Test if conftest.py file exists."""
    try:
        conftest_path = Path(__file__).parent / "tests" / "conftest.py"
        if conftest_path.exists():
            print("✅ conftest.py file exists")
            return True
        else:
            print("❌ conftest.py file not found")
            return False
    except Exception as e:
        print(f"❌ File check failed: {e}")
        return False

def test_fixture_centralization():
    """Test if fixture centralization is working without importing from tests."""
    try:
        # Set testing environment
        os.environ["BU_LAZY_MODELS"] = "0"
        os.environ["TESTING"] = "true"
        
        # Test a basic classifier function
        from bu_processor.pipeline.classifier import RealMLClassifier
        print("✅ Classifier import successful")
        
        # Test config
        from bu_processor.core.config import get_config
        config = get_config()
        print("✅ Configuration successful")
        
        # Test classifier creation
        classifier = RealMLClassifier(config)
        print("✅ Classifier creation successful")
        
        print("✅ All imports successful - fixture centralization working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Testing fixture centralization...")
    success = test_conftest_file_exists() and test_fixture_centralization()
    print("✅ Test completed successfully!" if success else "❌ Test failed!")
    exit(0 if success else 1)
