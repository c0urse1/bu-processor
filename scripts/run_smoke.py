#!/usr/bin/env python3
"""
CLI wrapper for smoke tests
===========================

Optional CLI script that calls pytest internally.
Provides backwards compatibility for direct script execution.
"""

import sys
import pytest
from pathlib import Path

def main():
    """Main entry point for smoke test CLI."""
    
    # Determine which tests to run based on arguments
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        test_type = "smoke"
    
    # Map test types to pytest arguments
    test_configs = {
        "quick": ["-q", "tests/smoke/test_quick.py", "--maxfail=1"],
        "comprehensive": ["-q", "tests/smoke/test_comprehensive.py", "--maxfail=1"], 
        "validate": ["-q", "tests/smoke/test_validate.py", "--maxfail=1"],
        "smoke": ["-q", "tests/smoke/", "--maxfail=1"],
        "all": ["-q", "--maxfail=1"]
    }
    
    # Get pytest arguments for the requested test type
    pytest_args = test_configs.get(test_type, test_configs["smoke"])
    
    print(f"Running {test_type} tests via pytest...")
    print(f"Command: pytest {' '.join(pytest_args)}")
    
    # Run pytest and return its exit code
    return pytest.main(pytest_args)

if __name__ == "__main__":
    sys.exit(main())
