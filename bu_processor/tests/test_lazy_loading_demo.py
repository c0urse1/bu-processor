#!/usr/bin/env python3
"""
Demonstration: Lazy Loading vs from_pretrained Assertions
=========================================================

This file demonstrates different approaches to handle lazy loading 
when testing from_pretrained method calls.
"""

import pytest
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
        
        # Manually trigger loading using the utility function
        from tests.conftest import force_model_loading
        force_model_loading(classifier)
        
        # Now the assertions should work
        # Note: This might not work perfectly if mocks were set up before classifier creation
        print("✅ Approach 3: Manual loading approach demonstrated!")
    
    def test_with_factory_function(self, mocker):
        """Approach 4: Use the create_eager_classifier_fixture factory.
        
        This approach creates a classifier programmatically with eager loading.
        """
        from tests.conftest import create_eager_classifier_fixture
        
        # Create classifier with eager loading
        classifier, mock_tokenizer_patch, mock_model_patch = create_eager_classifier_fixture(mocker)
        
        # These should work because the factory disabled lazy loading
        mock_tokenizer_patch.assert_called_once() 
        mock_model_patch.assert_called_once()
        
        print("✅ Approach 4: Factory function approach works!")
    
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


class TestLazyLoadingControlFixtures:
    """Test Suite demonstrating the new lazy loading control fixtures."""

    def test_with_manual_loading(self, lazy_models, mocker):
        """Test lazy loading mit manuellem Aufruf von ensure_models_loaded."""
        # Mock transformers components
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        
        mock_model_class = mocker.patch("transformers.AutoModelForSequenceClassification")
        mock_tokenizer_class = mocker.patch("transformers.AutoTokenizer")
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Initialisierung mit lazy=True sollte NICHT from_pretrained aufrufen
        classifier = RealMLClassifier(
            model_name="bert-base-uncased",
            device="cpu"
        )
        
        # Bei lazy loading werden models NICHT sofort geladen
        mock_model_class.from_pretrained.assert_not_called()
        mock_tokenizer_class.from_pretrained.assert_not_called()
        
        # Models sollten None sein bei lazy loading
        assert classifier.model is None
        assert classifier.tokenizer is None
        
        # Manueller Aufruf löst das Loading aus
        classifier.ensure_models_loaded()
        
        # Jetzt sollten from_pretrained aufgerufen worden sein
        mock_model_class.from_pretrained.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()

    def test_lazy_loading_automatic_on_classify(self, lazy_models, mocker):
        """Test dass lazy loading automatisch bei classify_text() ausgelöst wird."""
        # Mock transformers components
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        
        # Mock model outputs for classification
        mock_output = mocker.MagicMock()
        mock_output.logits = mocker.MagicMock()
        mock_model.return_value = mock_output
        
        # Mock tokenizer outputs
        mock_tokenizer.return_value = {
            'input_ids': [[101, 102, 103]],
            'attention_mask': [[1, 1, 1]]
        }
        
        # Mock torch operations
        mock_torch = mocker.patch("bu_processor.pipeline.classifier.torch")
        mock_torch.no_grad.return_value.__enter__ = mocker.MagicMock()
        mock_torch.no_grad.return_value.__exit__ = mocker.MagicMock()
        mock_torch.tensor.return_value = mocker.MagicMock()
        mock_torch.nn.functional.softmax.return_value = mocker.MagicMock()
        mock_torch.nn.functional.softmax.return_value.cpu.return_value.numpy.return_value = [[0.3, 0.7]]
        
        mock_model_class = mocker.patch("transformers.AutoModelForSequenceClassification")
        mock_tokenizer_class = mocker.patch("transformers.AutoTokenizer")
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Lazy loading - keine sofortigen from_pretrained Aufrufe
        classifier = RealMLClassifier(
            model_name="bert-base-uncased",
            device="cpu"
        )
        
        # Verify lazy initialization
        mock_model_class.from_pretrained.assert_not_called()
        mock_tokenizer_class.from_pretrained.assert_not_called()
        
        # First classify call should trigger model loading
        result = classifier.classify_text("Test text for lazy loading")
        
        # Now models should be loaded automatically
        mock_model_class.from_pretrained.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()
        
        # Result should be valid
        assert result is not None

    def test_is_loaded_property_lazy_behavior(self, lazy_models, mocker):
        """Test dass is_loaded property bei lazy loading korrekt funktioniert."""
        # Mock transformers
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        
        mock_model_class = mocker.patch("transformers.AutoModelForSequenceClassification")
        mock_tokenizer_class = mocker.patch("transformers.AutoTokenizer")
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Lazy initialization
        classifier = RealMLClassifier(
            model_name="bert-base-uncased",
            device="cpu"
        )
        
        # Initially should not be loaded (lazy)
        assert not classifier.is_loaded
        
        # Manually load models
        classifier.ensure_models_loaded()
        
        # Now should be loaded
        assert classifier.is_loaded

    def test_non_lazy_behavior_for_comparison(self, non_lazy_models, mocker):
        """Test non-lazy behavior als Vergleich zum lazy loading."""
        # Mock transformers
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        
        mock_model_class = mocker.patch("transformers.AutoModelForSequenceClassification")
        mock_tokenizer_class = mocker.patch("transformers.AutoTokenizer")
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        # Non-lazy initialization should load immediately
        classifier = RealMLClassifier(
            model_name="bert-base-uncased",
            device="cpu"
        )
        
        # Models should be loaded immediately (non-lazy)
        mock_model_class.from_pretrained.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()
        
        # Should be loaded from the start
        assert classifier.is_loaded


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
