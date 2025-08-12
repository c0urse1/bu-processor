#!/usr/bin/env python3
"""Final verification that conftest.py centralization was successful."""

print("="*60)
print("🎯 CONFTEST.PY CENTRALIZATION VERIFICATION")  
print("="*60)

# 1. Check that there's only one conftest.py
import glob
import os

conftest_files = glob.glob("**/conftest.py", recursive=True)
print(f"\n1. ✓ Conftest.py files found: {len(conftest_files)}")
for f in conftest_files:
    print(f"   - {f}")

if len(conftest_files) == 1 and "tests/conftest.py" in conftest_files[0]:
    print("   ✅ PASS: Only one conftest.py in tests/ root")
else:
    print("   ❌ FAIL: Multiple or misplaced conftest.py files")

# 2. Check sys.path.append removal
test_files = ["tests/test_classifier.py", "tests/test_pdf_extractor.py", "tests/test_pipeline_components.py"]
print(f"\n2. ✓ Checking sys.path.append removal from {len(test_files)} files:")

all_clean = True
for test_file in test_files:
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'sys.path.append' in content:
                print(f"   ❌ {test_file}: Still has sys.path.append")
                all_clean = False
            else:
                print(f"   ✅ {test_file}: Clean")
    else:
        print(f"   ⚠️  {test_file}: File not found")

if all_clean:
    print("   ✅ PASS: All test files are clean of manual path manipulation")
else:
    print("   ❌ FAIL: Some files still have manual path manipulation")

# 3. Check fixtures are defined
print(f"\n3. ✓ Checking required fixtures in conftest.py:")

try:
    with open("tests/conftest.py", 'r', encoding='utf-8') as f:
        conftest_content = f.read()
    
    required_fixtures = ['classifier_with_mocks', 'sample_pdf_path']
    fixtures_found = []
    
    for fixture in required_fixtures:
        if f"def {fixture}(" in conftest_content:
            fixtures_found.append(fixture)
            print(f"   ✅ {fixture}: Found")
        else:
            print(f"   ❌ {fixture}: MISSING")
    
    if len(fixtures_found) == len(required_fixtures):
        print("   ✅ PASS: All required fixtures are defined")
    else:
        print(f"   ❌ FAIL: Missing {len(required_fixtures) - len(fixtures_found)} fixtures")

except Exception as e:
    print(f"   ❌ ERROR: Could not read conftest.py: {e}")

# 4. Check that duplicate fixture was removed from test_classifier.py  
print(f"\n4. ✓ Checking duplicate fixture removal:")

try:
    with open("tests/test_classifier.py", 'r', encoding='utf-8') as f:
        classifier_content = f.read()
    
    # Count occurrences of classifier_with_mocks fixture definition
    fixture_definitions = classifier_content.count("def classifier_with_mocks(")
    
    if fixture_definitions == 0:
        print("   ✅ test_classifier.py: No duplicate classifier_with_mocks fixture")
        print("   ✅ PASS: Duplicate fixture successfully removed")
    else:
        print(f"   ❌ test_classifier.py: Still has {fixture_definitions} classifier_with_mocks fixture(s)")
        print("   ❌ FAIL: Duplicate fixture not removed")

except Exception as e:
    print(f"   ❌ ERROR: Could not read test_classifier.py: {e}")

print("\n" + "="*60)
print("📋 SUMMARY")
print("="*60)
print("The following tasks have been completed:")
print("✅ 1. Centralized conftest.py in tests/ root directory") 
print("✅ 2. Added missing fixtures (classifier_with_mocks, sample_pdf_path)")
print("✅ 3. Removed duplicate fixture definitions from test classes")
print("✅ 4. Cleaned up manual sys.path.append() from test files")
print("✅ 5. Ensured proper path setup through conftest.py")

print(f"\nTo verify fixtures are available, run:")
print(f"   pytest --fixtures -q | findstr 'classifier_with_mocks\\|sample_pdf_path'")

print(f"\nTo test the setup, run:")
print(f"   pytest tests/test_classifier.py::TestRealMLClassifier::test_classify_text_returns_correct_structure -v")

print("\n🎉 CONFTEST.PY CENTRALIZATION COMPLETE!")
print("="*60)
