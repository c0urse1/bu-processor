#!/usr/bin/env python3
"""Test Timeout/Retry fixes - Fix #10"""

import sys
from pathlib import Path
import time

# Add bu_processor to Python path
script_dir = Path(__file__).resolve().parent
bu_processor_dir = script_dir / "bu_processor"
sys.path.insert(0, str(bu_processor_dir))

def test_retry_decorator_functionality():
    """Test dass der Retry-Decorator korrekt funktioniert."""
    
    print("🔍 Testing Retry Decorator Functionality...")
    
    try:
        # Import required components
        from bu_processor.pipeline.classifier import (
            with_retry_and_timeout,
            ClassificationRetryError,
            ClassificationTimeout
        )
        print("✅ Successfully imported retry decorator and exceptions")
        
        # Test 1: Immediate success should not retry
        call_count = 0
        
        @with_retry_and_timeout(max_retries=3, base_delay=0.01)
        def immediate_success():
            nonlocal call_count
            call_count += 1
            return {"result": "success", "call_count": call_count}
        
        result = immediate_success()
        assert result["result"] == "success", "Function should succeed immediately"
        assert result["call_count"] == 1, "Function should only be called once"
        print("✅ Immediate success test passed")
        
        # Test 2: Eventual success after retries
        call_count = 0
        
        @with_retry_and_timeout(max_retries=3, base_delay=0.01)
        def eventual_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError(f"Failure attempt {call_count}")
            return {"result": "success after retries", "attempts": call_count}
        
        result = eventual_success()
        assert result["result"] == "success after retries", "Should succeed after retries"
        assert result["attempts"] == 3, "Should take 3 attempts"
        print("✅ Eventual success test passed")
        
        # Test 3: Max retries exceeded should raise ClassificationRetryError
        call_count = 0
        
        @with_retry_and_timeout(max_retries=2, base_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"Persistent failure {call_count}")
        
        try:
            always_fails()
            assert False, "Should have raised ClassificationRetryError"
        except ClassificationRetryError as e:
            assert "failed after 3 attempts" in str(e), "Error message should mention attempts"
            assert call_count == 3, "Should attempt max_retries + 1 times"
            print("✅ Max retries exceeded test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing retry decorator: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_timeout_functionality():
    """Test dass Timeout-Handling funktioniert."""
    
    print("\n🔍 Testing Timeout Functionality...")
    
    try:
        from bu_processor.pipeline.classifier import (
            with_retry_and_timeout,
            ClassificationTimeout
        )
        
        # Test timeout with mocked time
        original_time = time.time
        call_count = 0
        
        def mock_time():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 1000.0  # Start time
            else:
                return 1000.2  # End time (0.2s elapsed > 0.1s timeout)
        
        # Temporarily replace time.time
        time.time = mock_time
        
        try:
            @with_retry_and_timeout(timeout_seconds=0.1)
            def slow_function():
                # This function will appear to take 0.2s due to mocked time
                return "should not return"
            
            try:
                slow_function()
                assert False, "Should have raised ClassificationTimeout"
            except ClassificationTimeout as e:
                assert "exceeded timeout" in str(e), "Error should mention timeout"
                assert "0.1" in str(e), "Error should mention timeout duration"
                print("✅ Timeout test passed")
                
        finally:
            # Restore original time function
            time.time = original_time
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing timeout: {e}")
        return False

def test_mixed_exception_handling():
    """Test dass verschiedene Exception-Typen korrekt behandelt werden."""
    
    print("\n🔍 Testing Mixed Exception Handling...")
    
    try:
        from bu_processor.pipeline.classifier import (
            with_retry_and_timeout,
            ClassificationRetryError
        )
        
        call_count = 0
        exception_types = []
        
        @with_retry_and_timeout(max_retries=3, base_delay=0.01)
        def mixed_exceptions():
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                exc = ConnectionError("Network failure")
                exception_types.append(type(exc).__name__)
                raise exc
            elif call_count == 2:
                exc = RuntimeError("Runtime failure")
                exception_types.append(type(exc).__name__)
                raise exc
            elif call_count == 3:
                exc = ConnectionError("Another network failure")
                exception_types.append(type(exc).__name__)
                raise exc
            else:
                return {"result": "success", "handled_exceptions": exception_types}
        
        result = mixed_exceptions()
        assert result["result"] == "success", "Should succeed after handling mixed exceptions"
        assert len(result["handled_exceptions"]) == 3, "Should have handled 3 exceptions"
        assert call_count == 4, "Should take 4 calls (3 failures + 1 success)"
        print("✅ Mixed exception handling test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing mixed exceptions: {e}")
        return False

def test_decorator_parameters():
    """Test dass Decorator-Parameter korrekt funktionieren."""
    
    print("\n🔍 Testing Decorator Parameters...")
    
    try:
        from bu_processor.pipeline.classifier import with_retry_and_timeout
        
        # Test with different parameters
        @with_retry_and_timeout(
            max_retries=1,
            base_delay=0.01,
            timeout_seconds=5.0,
            backoff_factor=1.5
        )
        def parameterized_function():
            return {"result": "success with custom parameters"}
        
        result = parameterized_function()
        assert result["result"] == "success with custom parameters", "Custom parameters should work"
        print("✅ Custom parameters test passed")
        
        # Test with minimal parameters
        @with_retry_and_timeout()
        def default_parameters():
            return {"result": "success with defaults"}
        
        result = default_parameters()
        assert result["result"] == "success with defaults", "Default parameters should work"
        print("✅ Default parameters test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing decorator parameters: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Timeout/Retry Robustness Fixes (Fix #10)")
    print("=" * 60)
    
    success = True
    
    # Test all aspects
    success &= test_retry_decorator_functionality()
    success &= test_timeout_functionality()
    success &= test_mixed_exception_handling()
    success &= test_decorator_parameters()
    
    print("\n" + "=" * 60)
    if success:
        print("🎯 ALL TIMEOUT/RETRY TESTS PASSED!")
        print("✅ Fix #10: Timeout/Retry‑Test robust machen - COMPLETED")
        print("\n📋 Summary of implemented fixes:")
        print("   • Enhanced timeout handling with better error messages")
        print("   • Robust retry logic with proper exception handling")
        print("   • Improved test reliability with proper mocking strategies")
        print("   • Mixed exception type handling verified")
        print("   • Configurable decorator parameters tested")
        print("   • ClassificationRetryError and ClassificationTimeout working correctly")
        print("\n🎉 Timeout/Retry system is now robust and test-friendly!")
    else:
        print("❌ Some tests failed!")
    
    sys.exit(0 if success else 1)
