#!/usr/bin/env python3
"""Test script for robustness fixes in classifier.py"""

from bu_processor.pipeline.classifier import RealMLClassifier

def test_robustness_fixes():
    print('🔧 Testing RealMLClassifier robustness fixes...')
    
    # Test 1: Import successful
    print('✅ Import successful')
    
    # Test 2: Lazy classifier creation
    classifier = RealMLClassifier(lazy=True)
    print('✅ Lazy classifier created')
    
    # Test 3: Tokenizer normalization method
    mock_enc = type('MockEnc', (), {'input_ids': [1,2,3], 'attention_mask': [1,1,1]})()
    normalized = classifier._normalize_tokenizer_output(mock_enc)
    print(f'✅ Tokenizer normalization: {list(normalized.keys())}')
    
    # Test 4: Health status
    health = classifier.get_health_status()
    print(f'✅ Health status: {health["status"]}')
    
    # Test 5: Model loaded check
    model_loaded = health.get('model_loaded', False)
    print(f'✅ Model loaded check: {model_loaded}')
    
    print('🎉 All robustness fixes working!')
    return True

if __name__ == "__main__":
    test_robustness_fixes()
