#!/usr/bin/env python3
"""Simple test of enhanced features."""

from bu_processor.pipeline.classifier import RealMLClassifier

print('=== Testing Enhanced Features ===')

# Test lazy mode
c = RealMLClassifier(lazy=True)
print('✓ Lazy mode initialized')
print('is_loaded:', c.is_loaded)
print('has _load_lock:', hasattr(c, '_load_lock'))

# Test make_eager method exists
print('has make_eager method:', hasattr(c, 'make_eager'))

# Test ensure_models_loaded with force parameter
print('ensure_models_loaded callable:', callable(getattr(c, 'ensure_models_loaded', None)))

try:
    # Test the force parameter
    c.ensure_models_loaded(force=False)
    print('✓ ensure_models_loaded(force=False) works')
except Exception as e:
    print('Note: Loading failed (expected if no model):', type(e).__name__)

print('=== Enhanced Features Test Complete ===')
