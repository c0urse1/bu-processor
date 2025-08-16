#!/usr/bin/env python3
"""Debug script to test classifier integration scenario."""

import torch
from unittest.mock import Mock, patch
from bu_processor.pipeline.classifier import RealMLClassifier

def test_debug():
    """Debug the integration test to see what's happening."""
    
    # Set up mocks exactly like the test
    mock_tokenizer = Mock()
    mock_encoding = Mock()
    mock_encoding.input_ids = torch.tensor([[101, 2045, 2003, 102]])  # BERT-like token IDs
    mock_encoding.attention_mask = torch.tensor([[1, 1, 1, 1]])
    mock_encoding.to = Mock(return_value=mock_encoding)
    
    # Make the encoding behave like a dictionary for **kwargs unpacking
    mock_encoding.keys = Mock(return_value=['input_ids', 'attention_mask'])
    mock_encoding.__getitem__ = Mock(side_effect=lambda k: mock_encoding.input_ids if k == 'input_ids' else mock_encoding.attention_mask)
    mock_encoding.__iter__ = Mock(return_value=iter(['input_ids', 'attention_mask']))
    
    mock_tokenizer.return_value = mock_encoding
    
    mock_model = Mock()
    mock_outputs = Mock()
    # Realistische Logits für 3 Kategorien
    mock_outputs.logits = torch.tensor([[0.1, 0.85, 0.05]])  # Finance category (1) sehr wahrscheinlich
    
    def mock_model_call(*args, **kwargs):
        print(f"Model called with args={args}, kwargs={kwargs}")
        return mock_outputs
    
    mock_model.side_effect = mock_model_call
    mock_model.to = Mock()
    mock_model.eval = Mock()
    
    # Mock the labels from model config
    fake_config = Mock()
    fake_config.id2label = {0: "legal", 1: "finance", 2: "other"}
    mock_model.config = fake_config
    
    # Mock the _softmax method to return proper probabilities
    def mock_softmax(logits):
        print(f"Softmax called with logits: {logits}")
        result = [0.1, 0.85, 0.05]  # Finance category (1) most likely
        print(f"Softmax returning: {result}")
        return result
    
    with patch("bu_processor.pipeline.classifier.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("bu_processor.pipeline.classifier.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.device"), \
         patch("bu_processor.pipeline.classifier.EnhancedPDFExtractor"), \
         patch("bu_processor.pipeline.classifier.RealMLClassifier._softmax", side_effect=mock_softmax):
        
        classifier = RealMLClassifier()
        
        # Test realistischen Finanz-Text
        finance_text = "Unsere Quartalszahlen zeigen einen Umsatz von 2.5 Millionen Euro und einen Gewinn von 15%. Die Aktie ist gestiegen."
        
        print(f"Classifying text: {finance_text[:50]}...")
        
        # Debug _forward_logits
        try:
            print("Calling _forward_logits...")
            logits = classifier._forward_logits(finance_text)
            print(f"_forward_logits returned: {logits}")
        except Exception as e:
            print(f"Error in _forward_logits: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Debug _label_list
        try:
            print("Calling _label_list...")
            labels = classifier._label_list()
            print(f"_label_list returned: {labels}")
        except Exception as e:
            print(f"Error in _label_list: {e}")
            import traceback
            traceback.print_exc()
            return
            
        # Debug _postprocess_logits
        try:
            print("Calling _postprocess_logits...")
            result = classifier._postprocess_logits(logits, labels, finance_text)
            print(f"_postprocess_logits returned: {result}")
            
            if hasattr(result, 'dict'):
                result_data = result.dict()
            else:
                result_data = result
                
            print(f"Result data: {result_data}")
            print(f"Category: {result_data.get('category')}")
            
        except Exception as e:
            print(f"Error in _postprocess_logits: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Test full classification
        try:
            print("Calling full classify_text...")
            result = classifier.classify_text(finance_text)
            
            if hasattr(result, 'dict'):
                result_data = result.dict()
            else:
                result_data = result
                
            print(f"Full classification result: {result_data}")
            print(f"Category: {result_data.get('category')}")
            
        except Exception as e:
            print(f"Error in classify_text: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_debug()
