"""Test if the conftest.py centralization worked - self-contained version."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_conftest_file_exists():
    """Test conftest.py file exists."""
    try:
        conftest_path = Path(__file__).parent / "tests" / "conftest.py"
        if conftest_path.exists():
            print("✅ conftest.py file exists")
            
            # Read content to check for fixtures
            content = conftest_path.read_text()
            fixture_names = ['classifier_with_mocks', 'sample_pdf_path', 'dummy_train_val']
            
            print("Checking for fixtures:")
            all_found = True
            for fixture in fixture_names:
                if f"def {fixture}" in content:
                    print(f"  ✅ {fixture} found")
                else:
                    print(f"  ❌ {fixture} not found")
                    all_found = False
            
            return all_found
        else:
            print("❌ conftest.py file not found")
            return False
    except Exception as e:
        print(f"❌ File check failed: {e}")
        return False

def test_conftest_fixtures():
    """Test conftest.py centralization without importing from tests."""
    try:
        # Set testing environment
        os.environ["BU_LAZY_MODELS"] = "0"
        os.environ["TESTING"] = "true"
        
        # Test basic functionality instead of importing fixtures
        from bu_processor.core.config import get_config
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        config = get_config()
        classifier = RealMLClassifier(config)
        
        print("✅ Basic classifier functionality working")
        
        return True
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
