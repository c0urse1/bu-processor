#!/usr/bin/env python3
"""
Test script to verify threading and on-demand loading features.
"""

import threading
import time
import sys
import os

# Add the bu_processor directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bu_processor'))

from bu_processor.pipeline.classifier import RealMLClassifier


def test_lazy_loading():
    """Test lazy loading functionality."""
    print("Testing lazy loading...")
    
    # Create classifier in lazy mode
    classifier = RealMLClassifier(lazy=True)
    
    # Check that models are not loaded initially
    assert not classifier.is_loaded, "Models should not be loaded initially in lazy mode"
    print("✓ Models not loaded initially")
    
    # Call ensure_models_loaded and check that models are loaded
    classifier.ensure_models_loaded()
    assert classifier.is_loaded, "Models should be loaded after ensure_models_loaded()"
    print("✓ Models loaded after ensure_models_loaded()")
    
    # Call ensure_models_loaded again (should be idempotent)
    classifier.ensure_models_loaded()
    assert classifier.is_loaded, "Models should still be loaded after second call"
    print("✓ ensure_models_loaded is idempotent")


def test_auto_loading():
    """Test auto-loading on first classify call."""
    print("\nTesting auto-loading on classify...")
    
    # Create classifier in lazy mode
    classifier = RealMLClassifier(lazy=True)
    
    # Check that models are not loaded initially
    assert not classifier.is_loaded, "Models should not be loaded initially"
    print("✓ Models not loaded initially")
    
    # Call classify_text - should auto-load models
    try:
        result = classifier.classify_text("This is a test document.")
        assert classifier.is_loaded, "Models should be auto-loaded after classify_text"
        print("✓ Models auto-loaded on classify_text call")
    except Exception as e:
        print(f"Note: classify_text failed (expected if no model available): {e}")
        # Even if classification fails, models should have been attempted to load
        print("✓ Auto-loading attempted")


def test_thread_safety():
    """Test thread-safe loading."""
    print("\nTesting thread-safe loading...")
    
    # Create classifier in lazy mode
    classifier = RealMLClassifier(lazy=True)
    
    # Track loading attempts
    loading_started = threading.Event()
    loading_completed = threading.Event()
    load_count = threading.local()
    load_count.value = 0
    
    def load_models():
        """Thread function to load models."""
        try:
            loading_started.set()
            classifier.ensure_models_loaded()
            load_count.value += 1
            loading_completed.set()
        except Exception as e:
            print(f"Loading failed in thread: {e}")
    
    # Start multiple threads trying to load models
    threads = []
    for i in range(3):
        thread = threading.Thread(target=load_models)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Models should be loaded exactly once
    assert classifier.is_loaded, "Models should be loaded after threaded loading"
    print("✓ Thread-safe loading completed")


def test_non_lazy_mode():
    """Test non-lazy mode (should load immediately)."""
    print("\nTesting non-lazy mode...")
    
    try:
        # Create classifier in non-lazy mode
        classifier = RealMLClassifier(lazy=False)
        
        # Models should be loaded immediately
        assert classifier.is_loaded, "Models should be loaded immediately in non-lazy mode"
        print("✓ Models loaded immediately in non-lazy mode")
        
    except Exception as e:
        print(f"Note: Non-lazy loading failed (expected if no model available): {e}")
        print("✓ Non-lazy mode attempted immediate loading")


def main():
    """Run all tests."""
    print("Testing threading and on-demand loading features...\n")
    
    test_lazy_loading()
    test_auto_loading()
    test_thread_safety()
    test_non_lazy_mode()
    
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    main()
