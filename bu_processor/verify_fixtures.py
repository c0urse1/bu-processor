#!/usr/bin/env python3
"""Verify that conftest.py fixtures are working correctly - self-contained version."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_conftest_file_exists():
    """Check if conftest.py file exists."""
    try:
        conftest_path = project_root / "tests" / "conftest.py"
        if conftest_path.exists():
            print("✓ conftest.py file exists")
            return True
        else:
            print("✗ conftest.py file not found")
            return False
    except Exception as e:
        print(f"✗ file check failed: {e}")
        return False

def check_conftest_imports():
    """Check if conftest.py can be imported."""
    try:
        # Set testing environment
        os.environ["BU_LAZY_MODELS"] = "0"
        os.environ["TESTING"] = "true"
        
        # Import conftest directly
        conftest_path = project_root / "tests" / "conftest.py"
        spec = __import__('importlib.util', fromlist=['spec_from_file_location']).spec_from_file_location("conftest", conftest_path)
        conftest = __import__('importlib.util', fromlist=['module_from_spec']).module_from_spec(spec)
        spec.loader.exec_module(conftest)
        
        print("✓ conftest.py imports successfully")
        return True, conftest
    except Exception as e:
        print(f"✗ conftest.py import failed: {e}")
        return False, None

def check_fixtures(conftest_module):
    """Check if key fixtures are defined."""
    try:
        if not conftest_module:
            return False
            
        # Get all attributes from conftest
        fixtures = []
        for attr_name in dir(conftest_module):
            attr = getattr(conftest_module, attr_name)
            if hasattr(attr, '_pytestfixturefunction'):
                fixtures.append(attr_name)
        
        required_fixtures = ['classifier_with_mocks', 'sample_pdf_path', 'dummy_train_val']
        
        print("\nFound fixtures:")
        for fixture in fixtures:
            print(f"  - {fixture}")
        
        print(f"\nChecking required fixtures:")
        missing_fixtures = []
        for req_fixture in required_fixtures:
            if req_fixture in fixtures:
                print(f"✓ {req_fixture} found")
            else:
                print(f"✗ {req_fixture} missing")
                missing_fixtures.append(req_fixture)
        
        return len(missing_fixtures) == 0
        
    except Exception as e:
        print(f"✗ fixture check failed: {e}")
        return False

def main():
    """Main verification function."""
    print("=== Conftest.py Verification ===")
    
    success = True
    success &= check_conftest_file_exists()
    
    imports_ok, conftest_module = check_conftest_imports()
    success &= imports_ok
    
    if imports_ok:
        success &= check_fixtures(conftest_module)
    
    if success:
        print("\n✓ All checks passed! conftest.py is working correctly.")
    else:
        print("\n✗ Some checks failed. Please review the issues above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
