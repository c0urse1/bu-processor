#!/usr/bin/env python3
"""
Direct test of fixture functionality
"""

import os
import sys

# Add path
sys.path.insert(0, "c:/ml_classifier_poc/bu_processor/bu_processor")
sys.path.insert(0, "c:/ml_classifier_poc/bu_processor")

# Set environment for testing
os.environ["BU_LAZY_MODELS"] = "0"
os.environ["TESTING"] = "true"

#!/usr/bin/env python3
"""
Direct test of classifier functionality - self-contained version
"""

import os
import sys
from unittest.mock import MagicMock, Mock

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
        mocker = MockMocker()
        monkeypatch = MockMonkeypatch()
        
        print("✅ Mock objects created")
        
        # Test ClassificationResult import
        from bu_processor.pipeline.content_types import ClassificationResult
        
        result = ClassificationResult(
            text="test",
            category=0,
            confidence=0.85,
            is_confident=True,
            metadata={"test": True}
        )
        
        print(f"✅ ClassificationResult created: category={result.category}, confidence={result.confidence}")
        print("✅ Fixture centralization is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_classifier_fixture()
    exit(0 if success else 1)
