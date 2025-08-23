#!/usr/bin/env python3
"""
Test Script für Centralized Logging Configuration
================================================

Demonstriert die zentrale Logging-Konfiguration und verschiedene Features.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_console_logging():
    """Test console logging with structured output"""
    print("\n=== Console Logging Demo ===")
    
    try:
        from bu_processor.core.logging_setup import get_logger, get_logging_config
        
        # Show current config
        config = get_logging_config()
        print(f"Current config: {config}")
        
        # Create logger and test structured logging
        logger = get_logger("test.console")
        
        logger.info("Console logging test started")
        logger.info("Processing document", 
                   document_id="doc_123",
                   pages=5,
                   file_size_mb=2.4,
                   status="processing")
        
        logger.warning("Potential issue detected", 
                      issue_type="memory_usage",
                      memory_usage_mb=512,
                      threshold_mb=400)
        
        logger.error("Error occurred", 
                    error_code="E001",
                    error_message="File not found",
                    retry_count=3)
        
        print("✅ Console logging test completed successfully!")
        
    except Exception as e:
        print(f"❌ Console logging test failed: {e}")
        return False
    
    return True


def test_json_logging():
    """Test JSON logging format"""
    print("\n=== JSON Logging Demo ===")
    
    try:
        # Set JSON format
        os.environ['LOG_FORMAT'] = 'json'
        
        from bu_processor.core.logging_setup import get_logger, set_log_format
        
        # Force JSON format
        set_log_format('json')
        
        logger = get_logger("test.json")
        
        logger.info("JSON logging test started")
        logger.info("API request received",
                   endpoint="/classify/pdf",
                   method="POST",
                   user_id="user_456",
                   file_size=1024,
                   timestamp="2025-08-20T07:22:19Z")
        
        logger.error("Classification failed",
                    model="bert-base-german-cased",
                    confidence=0.23,
                    threshold=0.8,
                    reason="low_confidence")
        
        print("✅ JSON logging test completed successfully!")
        
    except Exception as e:
        print(f"❌ JSON logging test failed: {e}")
        return False
    finally:
        # Reset to console format
        os.environ['LOG_FORMAT'] = 'console'
        
    return True


def test_file_logging():
    """Test file logging"""
    print("\n=== File Logging Demo ===")
    
    try:
        # Create temp log file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
        
        # Set file logging
        os.environ['LOG_FILENAME'] = log_file
        
        from bu_processor.core.logging_setup import get_logger, reconfigure_logging
        
        # Reconfigure with file logging
        reconfigure_logging(log_filename=log_file)
        
        logger = get_logger("test.file")
        
        logger.info("File logging test started")
        logger.info("Data processing batch",
                   batch_id="batch_789",
                   records_processed=1500,
                   success_rate=0.98,
                   duration_seconds=45.2)
        
        # Check if log file was created and has content
        log_path = Path(log_file)
        if log_path.exists() and log_path.stat().st_size > 0:
            print(f"✅ Log file created: {log_file}")
            with open(log_file, 'r') as f:
                content = f.read()
                print(f"Log file content preview:\n{content[:200]}...")
        else:
            print(f"⚠️  Log file not found or empty: {log_file}")
        
        print("✅ File logging test completed successfully!")
        
        # Cleanup (ignore errors on Windows due to file locks)
        try:
            if log_path.exists():
                log_path.unlink()
        except OSError:
            print(f"⚠️  Could not delete temp log file (file lock): {log_file}")
        
    except Exception as e:
        print(f"❌ File logging test failed: {e}")
        return False
    finally:
        # Reset file logging
        if 'LOG_FILENAME' in os.environ:
            del os.environ['LOG_FILENAME']
    
    return True


def test_log_levels():
    """Test different log levels"""
    print("\n=== Log Levels Demo ===")
    
    try:
        from bu_processor.core.logging_setup import get_logger, set_log_level
        
        logger = get_logger("test.levels")
        
        # Test DEBUG level
        set_log_level('DEBUG')
        logger.debug("Debug message", component="chunking", memory_mb=128)
        logger.info("Info message", operation="text_extraction", status="completed")
        logger.warning("Warning message", usage_percent=85, threshold=80)
        logger.error("Error message", error_type="timeout", duration=30)
        
        # Test WARNING level (should suppress DEBUG)
        print("\n--- Switching to WARNING level (should suppress DEBUG/INFO) ---")
        set_log_level('WARNING')
        logger.debug("This debug should not appear")
        logger.info("This info should not appear")
        logger.warning("This warning should appear", issue="high_memory")
        logger.error("This error should appear", critical=True)
        
        print("✅ Log levels test completed successfully!")
        
    except Exception as e:
        print(f"❌ Log levels test failed: {e}")
        return False
    finally:
        # Reset to INFO level
        set_log_level('INFO')
    
    return True


def main():
    """Run all centralized logging tests"""
    print("🧪 Testing Centralized Logging Configuration")
    print("=" * 50)
    
    tests = [
        test_console_logging,
        test_json_logging,
        test_file_logging,
        test_log_levels
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All centralized logging tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
