#!/usr/bin/env python3
"""Verify that conftest.py fixtures are working correctly."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_conftest_imports():
    """Check if conftest.py can be imported."""
    try:
        from tests.conftest import pytest_configure
        print("✓ conftest.py imports successfully")
        return True
    except Exception as e:
        print(f"✗ conftest.py import failed: {e}")
        return False

def check_fixtures():
    """Check if key fixtures are defined."""
    try:
        import pytest
        from tests import conftest
        
        # Get all attributes from conftest
        fixtures = [attr for attr in dir(conftest) 
                   if hasattr(getattr(conftest, attr), '_pytestfixturefunction')]
        
        required_fixtures = ['classifier_with_mocks', 'sample_pdf_path']
        
        print("\nFound fixtures:")
        for fixture in fixtures:
            print(f"  - {fixture}")
        
        print(f"\nChecking required fixtures:")
        for req_fixture in required_fixtures:
            if req_fixture in fixtures:
                print(f"✓ {req_fixture} found")
            else:
                print(f"✗ {req_fixture} missing")
        
        return all(req in fixtures for req in required_fixtures)
        
    except Exception as e:
        print(f"✗ fixture check failed: {e}")
        return False

def main():
    """Main verification function."""
    print("=== Conftest.py Verification ===")
    
    success = True
    success &= check_conftest_imports()
    success &= check_fixtures()
    
    if success:
        print("\n✓ All checks passed! conftest.py is working correctly.")
    else:
        print("\n✗ Some checks failed. Please review the issues above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
