#!/usr/bin/env python3
"""Debug the specific failing test."""

import torch
from unittest.mock import Mock, patch
from bu_processor.pipeline.classifier import RealMLClassifier

def debug_test_classify_returns_probabilities():
    """Debug the specific failing test."""
    
    # Mock device (same as test)
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.device", return_value="cpu"):
        
        # Mock Transformer components first, before class instantiation
        mock_tokenizer = Mock()
        mock_encoding = Mock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3]])
        mock_encoding.attention_mask = torch.tensor([[1, 1, 1]])
        mock_encoding.to = Mock(return_value=mock_encoding)
        
        # Make the encoding behave like a dictionary for **kwargs unpacking
        mock_encoding.keys = Mock(return_value=['input_ids', 'attention_mask'])
        mock_encoding.__getitem__ = Mock(side_effect=lambda k: mock_encoding.input_ids if k == 'input_ids' else mock_encoding.attention_mask)
        mock_encoding.__iter__ = Mock(return_value=iter(['input_ids', 'attention_mask']))
        
        mock_tokenizer.return_value = mock_encoding
        
        mock_model = Mock()
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[0.1, 5.0, 0.1]])  # Strong logits → high confidence
        mock_model.return_value = mock_outputs
        # Configure the mock to return outputs when called with **kwargs
        def mock_model_call(*args, **kwargs):
            print(f"Model called with args={args}, kwargs={kwargs}")
            return mock_outputs
        mock_model.side_effect = mock_model_call
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock()
        
        # Mock the labels from model config
        fake_config = Mock()
        fake_config.id2label = {0: "category_0", 1: "category_1", 2: "category_2"}
        mock_model.config = fake_config
        
        with patch("bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model), \
             patch("bu_processor.pipeline.classifier.EnhancedPDFExtractor"):
            
            # Test classifier creation und operation
            classifier = RealMLClassifier()
            
            # Mock the classifier's custom softmax method after instantiation
            with patch.object(classifier, '_softmax', return_value=[0.01, 0.99, 0.01]) as mock_softmax:
                
                print("Calling classify_text...")
                try:
                    result = classifier.classify_text("Test")
                    
                    print(f"Result: {result}")
                    
                    if hasattr(result, 'dict'):
                        result_data = result.dict()
                    else:
                        result_data = result
                    
                    print(f"Result data: {result_data}")
                    print(f"Category: {result_data.get('category')}")
                    
                    # Verify mocked softmax operation was called
                    print(f"Softmax called: {mock_softmax.called}")
                    print(f"Softmax call count: {mock_softmax.call_count}")
                    
                except Exception as e:
                    print(f"Error during classification: {e}")
                    import traceback
                    traceback.print_exc()

if __name__ == "__main__":
    debug_test_classify_returns_probabilities()
