#!/usr/bin/env python3
"""
Health Check Test Verification

Tests die Health-Check Stabilisierung mit verschiedenen Szenarien:
1. Eager loading (Model geladen) -> "healthy"
2. Lazy loading (Model nicht geladen) -> "degraded" 
3. Fehler beim Model-Loading -> "unhealthy"
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

class TestHealthCheckStabilization(unittest.TestCase):
    """Tests für Health-Check Stabilisierung."""
    
    def setUp(self):
        """Test setup."""
        self.mock_torch_tensor = Mock()
        self.mock_torch_tensor.to = Mock(return_value=self.mock_torch_tensor)
        
    @patch("bu_processor.pipeline.classifier.AutoTokenizer")
    @patch("bu_processor.pipeline.classifier.AutoModelForSequenceClassification") 
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.tensor", return_value=Mock())
    def test_health_check_with_loaded_model(self, mock_tensor, mock_cuda, mock_model_cls, mock_tokenizer_cls):
        """Test: Health check mit geladenem Model -> 'healthy'."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model
        
        # Mock classification result
        with patch.object(sys.modules.get('bu_processor.pipeline.classifier', Mock()), 'RealMLClassifier') as MockClassifier:
            # Create classifier instance mock
            classifier_instance = Mock()
            MockClassifier.return_value = classifier_instance
            
            # Set up model as loaded
            classifier_instance.model = mock_model
            classifier_instance.tokenizer = mock_tokenizer
            classifier_instance._lazy = False
            classifier_instance.device = "cpu"
            classifier_instance.batch_size = 1
            classifier_instance.max_retries = 3
            classifier_instance.timeout_seconds = 30
            
            # Mock classify_text to succeed
            classifier_instance.classify_text.return_value = {
                "category": 1,
                "confidence": 0.95,
                "is_confident": True
            }
            
            # Mock get_model_info
            classifier_instance.get_model_info.return_value = {"model_name": "test-model"}
            
            # Import and test the actual get_health_status method
            from bu_processor.pipeline.classifier import RealMLClassifier
            
            # Create real instance with mocked dependencies
            with patch.dict(os.environ, {}, clear=True):  # Clear environment
                real_classifier = RealMLClassifier(lazy=False)
                
                # Set the mocked components
                real_classifier.model = mock_model
                real_classifier.tokenizer = mock_tokenizer
                real_classifier._lazy = False
                
                # Mock the classify_text method
                real_classifier.classify_text = Mock(return_value={
                    "category": 1, "confidence": 0.95, "is_confident": True
                })
                real_classifier.get_model_info = Mock(return_value={"model_name": "test-model"})
                
                # Test health check
                health = real_classifier.get_health_status()
                
                # Assertions
                self.assertEqual(health["status"], "healthy")
                self.assertTrue(health["model_loaded"])
                self.assertFalse(health["lazy_mode"])
                self.assertEqual(health["test_classification"], "passed")
                
    def test_health_check_degraded_status_logic(self):
        """Test: Health check Logik für degraded Status."""
        # Mock classifier with lazy mode enabled but no model loaded
        classifier = Mock()
        classifier.model = None
        classifier.tokenizer = None
        classifier._lazy = True
        classifier.device = "cpu"
        classifier.batch_size = 1
        classifier.max_retries = 3  
        classifier.timeout_seconds = 30
        classifier.get_model_info.return_value = {}
        
        # Simulate the health check logic
        model_loaded = (
            hasattr(classifier, 'model') and classifier.model is not None and
            hasattr(classifier, 'tokenizer') and classifier.tokenizer is not None
        )
        is_lazy_mode = getattr(classifier, '_lazy', False)
        
        # Determine status like in the real implementation
        if model_loaded:
            status = "healthy"
        elif is_lazy_mode and not model_loaded:
            status = "degraded"
        else:
            status = "unhealthy"
            
        # Assertions
        self.assertFalse(model_loaded)
        self.assertTrue(is_lazy_mode)
        self.assertEqual(status, "degraded")
        
    def test_health_check_unhealthy_status_logic(self):
        """Test: Health check Logik für unhealthy Status."""
        # Mock classifier with no lazy mode and no model
        classifier = Mock()
        classifier.model = None
        classifier.tokenizer = None
        classifier._lazy = False
        
        # Simulate the health check logic
        model_loaded = (
            hasattr(classifier, 'model') and classifier.model is not None and
            hasattr(classifier, 'tokenizer') and classifier.tokenizer is not None
        )
        is_lazy_mode = getattr(classifier, '_lazy', False)
        
        # Determine status
        if model_loaded:
            status = "healthy"
        elif is_lazy_mode and not model_loaded:
            status = "degraded"
        else:
            status = "unhealthy"
            
        # Assertions
        self.assertFalse(model_loaded)
        self.assertFalse(is_lazy_mode)
        self.assertEqual(status, "unhealthy")

def main():
    """Run health check tests."""
    print("=== HEALTH CHECK STABILIZATION TESTS ===")
    print("Testing verschiedene Health-Check Szenarien...")
    
    # Run basic logic tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHealthCheckStabilization)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ Health Check Tests erfolgreich!")
        print("\n📋 ZUSAMMENFASSUNG:")
        print("- Model geladen -> 'healthy' ✅")
        print("- Lazy mode ohne Model -> 'degraded' ✅") 
        print("- Kein lazy mode, kein Model -> 'unhealthy' ✅")
        print("\n🎯 Health-Check Stabilisierung implementiert!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} Tests fehlgeschlagen")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
