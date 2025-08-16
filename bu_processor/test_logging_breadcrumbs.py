#!/usr/bin/env python3
"""
Test script to verify Pinecone logging breadcrumbs are working correctly.
This script tests all the logging scenarios described in point 6.
"""

import os
import logging
import sys
from io import StringIO
from contextlib import contextmanager

# Set up logging to capture messages
@contextmanager
def capture_logs():
    """Capture log messages for testing."""
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger("pinecone_integration")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    try:
        yield log_capture
    finally:
        logger.removeHandler(handler)

def test_logging_breadcrumbs():
    """Test all Pinecone logging breadcrumb scenarios."""
    print("🧪 Testing Pinecone logging breadcrumbs...")
    
    # Test 1: Stub mode with ALLOW_EMPTY_PINECONE_KEY=1
    print("\n📋 Test 1: Stub mode with ALLOW_EMPTY_PINECONE_KEY=1")
    os.environ['ALLOW_EMPTY_PINECONE_KEY'] = '1'
    os.environ.pop('PINECONE_API_KEY', None)  # Remove API key
    
    with capture_logs() as log_capture:
        from bu_processor.pipeline.pinecone_integration import get_pinecone_manager
        manager = get_pinecone_manager()
        results = manager.search_similar_documents("test query", top_k=3)
        
        logs = log_capture.getvalue()
        print(f"Manager type: {type(manager).__name__}")
        print(f"Results: {len(results)} items")
        print(f"Log messages captured: {len(logs.split('WARNING')) - 1} warnings")
        
        # Check for expected log patterns
        expected_patterns = [
            "running in STUB MODE",
            "no network calls"
        ]
        
        found_patterns = []
        for pattern in expected_patterns:
            if pattern.lower() in logs.lower():
                found_patterns.append(pattern)
        
        print(f"✅ Found expected patterns: {found_patterns}")
        if logs.strip():
            print(f"📝 Sample log content:\n{logs[:200]}...")
    
    # Test 2: Stub mode without ALLOW_EMPTY_PINECONE_KEY (missing API key)
    print("\n📋 Test 2: Stub mode without ALLOW_EMPTY_PINECONE_KEY")
    os.environ.pop('ALLOW_EMPTY_PINECONE_KEY', None)
    os.environ.pop('PINECONE_API_KEY', None)
    
    # Clear import cache to force re-initialization
    if 'bu_processor.pipeline.pinecone_integration' in sys.modules:
        del sys.modules['bu_processor.pipeline.pinecone_integration']
    
    with capture_logs() as log_capture:
        from bu_processor.pipeline.pinecone_integration import get_pinecone_manager
        manager = get_pinecone_manager()
        results = manager.search_similar_documents("test query 2", top_k=2)
        
        logs = log_capture.getvalue()
        print(f"Manager type: {type(manager).__name__}")
        print(f"Results: {len(results)} items")
        
        # Check for expected warning message
        expected_warning = "Pinecone not available or API key missing. Using STUB mode."
        if expected_warning in logs:
            print(f"✅ Found expected warning: '{expected_warning}'")
        else:
            print(f"⚠️ Expected warning not found in logs")
        
        if logs.strip():
            print(f"📝 Log content:\n{logs}")
    
    # Test 3: Pipeline integration logging
    print("\n📋 Test 3: Pipeline integration logging")
    os.environ['ALLOW_EMPTY_PINECONE_KEY'] = '1'
    
    with capture_logs() as log_capture:
        from bu_processor.pipeline.enhanced_integrated_pipeline import EnhancedIntegratedPipeline
        pipeline = EnhancedIntegratedPipeline()
        
        logs = log_capture.getvalue()
        print(f"Pipeline Pinecone manager: {type(pipeline.pinecone).__name__ if pipeline.pinecone else 'None'}")
        
        # Check for initialization logs
        if "STUB MODE" in logs or "stub mode" in logs.lower():
            print("✅ Found stub mode initialization logs")
        else:
            print("⚠️ No stub mode logs found in pipeline initialization")
        
        if logs.strip():
            print(f"📝 Pipeline logs preview:\n{logs[:300]}...")

    print("\n🎉 Logging breadcrumb test completed!")
    print("\nExpected behaviors:")
    print("- When ALLOW_EMPTY_PINECONE_KEY=1: Debug level logs for test mode")
    print("- When no API key and no test flag: Warning level logs")
    print("- Consistent messages across all manager types")

if __name__ == "__main__":
    test_logging_breadcrumbs()
