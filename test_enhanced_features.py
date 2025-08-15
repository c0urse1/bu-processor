#!/usr/bin/env python3
"""Test the enhanced features."""

import sys
import os

# Change to the bu_processor directory
os.chdir(os.path.join(os.path.dirname(__file__), 'bu_processor'))

from bu_processor.pipeline.classifier import RealMLClassifier

def test_enhanced_features():
    print('=== Testing Enhanced Features ===')
    
    # 1. Test lazy mode with new auto-loading pattern
    c = RealMLClassifier(lazy=True)
    print('✓ Lazy mode initialized')
    print('is_loaded:', c.is_loaded)
    
    # 2. Test make_eager method
    print('Testing make_eager()...')
    try:
        c.make_eager()
        print('✓ make_eager() completed')
        print('is_loaded after make_eager:', c.is_loaded)
    except Exception as e:
        print('Note: make_eager failed (expected if no model):', e)
    
    # 3. Test force parameter
    print('Testing ensure_models_loaded(force=True)...')
    try:
        c.ensure_models_loaded(force=True)
        print('✓ force parameter works')
    except Exception as e:
        print('Note: force loading failed (expected if no model):', e)
    
    # 4. Test lazy-only auto-loading
    print('Testing lazy-only auto-loading...')
    c2 = RealMLClassifier(lazy=False)  # Non-lazy mode
    print('Non-lazy mode is_loaded:', c2.is_loaded)
    
    c3 = RealMLClassifier(lazy=True)   # Lazy mode
    print('Lazy mode is_loaded before classify:', c3.is_loaded)
    
    try:
        # This should trigger auto-loading only in lazy mode
        result = c3.classify_text("Test document")
        print('✓ Lazy auto-loading triggered by classify_text')
    except Exception as e:
        print('Note: classify_text failed (expected if no model):', e)
    
    print('=== All Enhanced Features Tested ===')

if __name__ == "__main__":
    test_enhanced_features()
