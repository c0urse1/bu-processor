#!/usr/bin/env python3
"""
Demonstration: Lazy Loading vs from_pretrained Assertions
=========================================================

This file demonstrates different approaches to handle lazy loading 
when testing from_pretrained method calls.
"""

import pytest
import os
from unittest.mock import Mock
import torch


class TestLazyLoadingApproaches:
    """Demonstrate different approaches for lazy loading in tests."""
    
    def test_with_disable_lazy_loading_fixture(self, mocker, disable_lazy_loading):
        """Approach 1: Use the disable_lazy_loading fixture.
        
        This is the cleanest approach for tests that need to assert
        on from_pretrained calls immediately after initialization.
        """
        # Mock components
        mock_tokenizer = mocker.Mock()
        mock_model = mocker.Mock()
        mock_model.to = mocker.Mock(return_value=mock_model)
        mock_model.eval = mocker.Mock()
        
        # Patch the imports
        mock_tokenizer_patch = mocker.patch(
            "bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", 
            return_value=mock_tokenizer
        )
        mock_model_patch = mocker.patch(
            "bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", 
            return_value=mock_model
        )
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("bu_processor.pipeline.classifier.EnhancedPDFExtractor")
        
        # Create classifier - from_pretrained should be called immediately
        from bu_processor.pipeline.classifier import RealMLClassifier
        classifier = RealMLClassifier()
        
        # These assertions should work because lazy loading is disabled
        mock_tokenizer_patch.assert_called_once()
        mock_model_patch.assert_called_once()
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()
        
        print("✅ Approach 1: disable_lazy_loading fixture works!")
    
    def test_with_eager_loading_fixture(self, classifier_with_eager_loading):
        """Approach 2: Use the classifier_with_eager_loading fixture.
        
        This fixture creates a classifier with lazy loading disabled
        and provides access to the mock objects for assertions.
        """
        classifier = classifier_with_eager_loading
        
        # The fixture already created the classifier and stores mock references
        mock_tokenizer_patch = classifier._test_mock_tokenizer_patch
        mock_model_patch = classifier._test_mock_model_patch
        mock_model = classifier._test_mock_model
        
        # These should work because the fixture disabled lazy loading
        mock_tokenizer_patch.assert_called_once()
        mock_model_patch.assert_called_once()
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()
        
        print("✅ Approach 2: classifier_with_eager_loading fixture works!")
    
    def test_with_manual_loading(self, mocker, classifier_with_mocks):
        """Approach 3: Use manual loading with force_model_loading utility.
        
        This approach uses the regular classifier_with_mocks fixture
        but manually triggers model loading when needed.
        """
        # Get the classifier (may be lazy-loaded)
        classifier = classifier_with_mocks
        
        # Patch and store references
        mock_tokenizer_patch = mocker.patch(
            "bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained",
            return_value=mocker.Mock()
        )
        mock_model_patch = mocker.patch(
            "bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained",
            return_value=mocker.Mock()
        )
        
        # Manually trigger loading using environment variable
        os.environ["BU_LAZY_MODELS"] = "0"  # Force eager loading
        
        # Recreate classifier with eager loading
        from bu_processor.core.config import get_config
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        config = get_config()
        classifier = RealMLClassifier(config)
        
        print("✅ Approach 3: Manual loading approach demonstrated!")
    
    def test_with_factory_function(self, mocker):
        """Approach 4: Use environment variables for eager loading.
        
        This approach creates a classifier programmatically with eager loading.
        """
        # Set environment for eager loading
        os.environ["BU_LAZY_MODELS"] = "0"
        
        # Set up mocks
        mock_tokenizer_patch = mocker.patch(
            "bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained",
            return_value=mocker.Mock()
        )
        mock_model_patch = mocker.patch(
            "bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained",
            return_value=mocker.Mock()
        )
        
        # Create classifier with eager loading
        from bu_processor.core.config import get_config
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        config = get_config()
        classifier = RealMLClassifier(config)
        
        # These should work because eager loading is enabled
        mock_tokenizer_patch.assert_called_once() 
        mock_model_patch.assert_called_once()
        
        print("✅ Approach 4: Environment variable approach works!")
    
    def test_lazy_loading_behavior_default(self, mocker, classifier_with_mocks):
        """Demonstration: Default lazy loading behavior.
        
        This shows what happens with default lazy loading enabled.
        """
        classifier = classifier_with_mocks
        
        # With lazy loading (default), from_pretrained might not be called yet
        # So we shouldn't assert on them immediately after creation
        
        # Instead, trigger actual usage to force loading
        try:
            result = classifier.classify_text("test text")
            print("✅ Classifier works with lazy loading enabled")
        except Exception as e:
            print(f"⚠️  Classifier usage failed (expected with mocks): {e}")
        
        print("✅ Lazy loading behavior demonstrated!")
    
    def test_monkeypatch_approach(self, mocker, monkeypatch):
        """Approach 5: Direct monkeypatch in test.
        
        This shows how to disable lazy loading directly in a test
        without using a separate fixture.
        """
        # Disable lazy loading for this test
        monkeypatch.setenv("BU_LAZY_MODELS", "0")
        
        # Mock components
        mock_tokenizer = mocker.Mock()
        mock_model = mocker.Mock()
        mock_model.to = mocker.Mock(return_value=mock_model)
        mock_model.eval = mocker.Mock()
        
        # Patch the imports
        mock_tokenizer_patch = mocker.patch(
            "bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", 
            return_value=mock_tokenizer
        )
        mock_model_patch = mocker.patch(
            "bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", 
            return_value=mock_model
        )
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("bu_processor.pipeline.classifier.EnhancedPDFExtractor")
        
        # Create classifier
        from bu_processor.pipeline.classifier import RealMLClassifier
        classifier = RealMLClassifier()
        
        # These should work
        mock_tokenizer_patch.assert_called_once()
        mock_model_patch.assert_called_once()
        
        print("✅ Approach 5: Direct monkeypatch approach works!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
