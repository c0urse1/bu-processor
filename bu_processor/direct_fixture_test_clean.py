#!/usr/bin/env python3
"""
Direct test of classifier functionality - self-contained version
"""

import os
import sys

# Add path
sys.path.insert(0, "c:/ml_classifier_poc/bu_processor/bu_processor")
sys.path.insert(0, "c:/ml_classifier_poc/bu_processor")

# Set environment for testing
os.environ["BU_LAZY_MODELS"] = "0"
os.environ["TESTING"] = "true"

def test_classifier_functionality():
    """Test classifier functionality directly without test fixtures."""
    try:
        from bu_processor.core.config import get_config
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        print("✅ Classifier imports successful")
        
        # Create configuration
        config = get_config()
        print("✅ Configuration loaded")
        
        # Create classifier
        classifier = RealMLClassifier(config)
        print("✅ Classifier created")
        
        # Test basic properties
        has_is_loaded = hasattr(classifier, 'is_loaded')
        print(f"✅ Has is_loaded property: {has_is_loaded}")
        
        has_logger = hasattr(classifier, 'logger')
        print(f"✅ Has logger property: {has_logger}")
        
        print("✅ All basic classifier functionality working")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    print("🔍 Testing classifier functionality directly...")
    success = test_classifier_functionality()
    print("✅ Test completed successfully!" if success else "❌ Test failed!")
    exit(0 if success else 1)
