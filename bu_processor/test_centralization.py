"""Test if the conftest.py centralization worked."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_conftest_fixtures():
    """Test conftest.py fixtures are available."""
    try:
        # Import the conftest module
        from tests import conftest
        
        # Check if our fixtures exist
        fixtures_found = []
        
        # Get all functions/fixtures from conftest
        for name in dir(conftest):
            obj = getattr(conftest, name)
            if hasattr(obj, '_pytestfixturefunction'):
                fixtures_found.append(name)
        
        print("Available fixtures:")
        for fixture in sorted(fixtures_found):
            print(f"  - {fixture}")
            
        # Check required fixtures
        required = ['classifier_with_mocks', 'sample_pdf_path']
        missing = []
        
        for req in required:
            if req in fixtures_found:
                print(f"✓ Required fixture '{req}' found")
            else:
                missing.append(req)
                print(f"✗ Required fixture '{req}' MISSING")
        
        if not missing:
            print("\n✅ ALL REQUIRED FIXTURES FOUND!")
            return True
        else:
            print(f"\n❌ Missing fixtures: {missing}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing conftest: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_sys_path_append():
    """Test that sys.path.append was removed from test files."""
    test_files = [
        "tests/test_classifier.py",
        "tests/test_pdf_extractor.py", 
        "tests/test_pipeline_components.py"
    ]
    
    clean_files = []
    problematic_files = []
    
    for test_file in test_files:
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'sys.path.append' in content:
                    problematic_files.append(test_file)
                else:
                    clean_files.append(test_file)
    
    print(f"\nFiles without sys.path.append: {len(clean_files)}")
    for f in clean_files:
        print(f"  ✓ {f}")
        
    if problematic_files:
        print(f"\nFiles still with sys.path.append: {len(problematic_files)}")
        for f in problematic_files:
            print(f"  ✗ {f}")
        return False
    else:
        print("\n✅ All test files are clean of manual sys.path.append!")
        return True

def main():
    print("=== Testing conftest.py Centralization ===\n")
    
    success1 = test_conftest_fixtures()
    success2 = test_no_sys_path_append()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED! conftest.py centralization is complete.")
        return True
    else:
        print("\n💥 Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
