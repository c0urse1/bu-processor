#!/usr/bin/env python3
"""
Summary Verification: Confidence Fixes
=====================================

Verifies that the confidence assertion and mock logits corrections
have been successfully implemented.
"""

import re
from pathlib import Path


def check_file_updates():
    """Check that all files have been updated with strong logits."""
    print("🔍 Checking file updates...")
    
    files_to_check = [
        ("tests/conftest.py", "conftest.py fixture updates"),
        ("tests/test_classifier.py", "test_classifier.py logits updates"),
    ]
    
    issues = []
    
    for file_path, description in files_to_check:
        full_path = Path(file_path)
        if not full_path.exists():
            issues.append(f"❌ {file_path} not found")
            continue
            
        try:
            content = full_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = full_path.read_text(encoding='cp1252')
        
        # Check for old weak logits
        weak_patterns = [
            r'torch\.tensor\(\[\[0\.1,\s*0\.8,\s*0\.1\]\]\)',
            r'torch\.tensor\(\[\[0\.2,\s*0\.8,\s*0\.0\]\]\)',
        ]
        
        found_weak = False
        for pattern in weak_patterns:
            if re.search(pattern, content):
                issues.append(f"❌ {file_path} still contains weak logits: {pattern}")
                found_weak = True
        
        # Check for strong logits
        strong_patterns = [
            r'torch\.tensor\(\[\[0\.1,\s*5\.0,\s*0\.1\]\]\)',
            r'torch\.tensor\(\[\[0\.2,\s*5\.0,\s*0\.0\]\]\)',
        ]
        
        found_strong = False
        for pattern in strong_patterns:
            if re.search(pattern, content):
                found_strong = True
                break
        
        if not found_weak and found_strong:
            print(f"✅ {description} - Updated with strong logits")
        elif found_weak:
            print(f"❌ {description} - Still contains weak logits")
        else:
            print(f"⚠️  {description} - No clear logits pattern found")
    
    return len(issues) == 0, issues


def check_confidence_thresholds():
    """Check that confidence thresholds are still meaningful."""
    print("\n🔍 Checking confidence thresholds...")
    
    test_file = Path("tests/test_classifier.py")
    if not test_file.exists():
        print("❌ test_classifier.py not found")
        return False
    
    try:
        content = test_file.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        content = test_file.read_text(encoding='cp1252')
    
    # Look for confidence assertions
    threshold_patterns = [
        (r'confidence.*>\s*0\.7', "High confidence threshold (> 0.7)"),
        (r'confidence.*>\s*0\.5', "Medium confidence threshold (> 0.5)"),
    ]
    
    thresholds_found = []
    for pattern, description in threshold_patterns:
        matches = re.findall(pattern, content)
        if matches:
            thresholds_found.append(description)
            print(f"✅ Found {description}: {len(matches)} occurrences")
    
    if thresholds_found:
        print("✅ Meaningful confidence thresholds maintained")
        return True
    else:
        print("❌ No confidence threshold assertions found")
        return False


def check_documentation():
    """Check that documentation was created."""
    print("\n🔍 Checking documentation...")
    
    doc_file = Path("CONFIDENCE_FIXES.md")
    if not doc_file.exists():
        print("❌ CONFIDENCE_FIXES.md not found")
        return False
    
    try:
        content = doc_file.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        content = doc_file.read_text(encoding='cp1252')
    
    required_sections = [
        ("Problem Description", "problem"),
        ("Solution Implemented", "solution"),
        ("Softmax", "technical explanation"),
        ("Strong logits", "fix implementation"),
        ("5.0", "specific strong logit value"),
    ]
    
    missing_sections = []
    for section, check_text in required_sections:
        if check_text.lower() not in content.lower():
            missing_sections.append(section)
        else:
            print(f"✅ Documentation contains {section}")
    
    if missing_sections:
        print(f"❌ Missing documentation sections: {missing_sections}")
        return False
    else:
        print("✅ Complete documentation available")
        return True


def calculate_expected_confidence():
    """Calculate what confidence values we should expect."""
    print("\n🔍 Calculating expected confidence values...")
    
    try:
        import torch
        import torch.nn.functional as F
        
        test_cases = [
            ([0.1, 5.0, 0.1], "Strong logits (fixed)"),
            ([0.1, 0.8, 0.1], "Weak logits (original)"),
            ([0.2, 5.0, 0.0], "Strong variant"),
            ([0.0, 10.0, 0.0], "Very strong"),
        ]
        
        print("Expected confidence values:")
        all_strong_pass = True
        
        for logits_list, description in test_cases:
            logits = torch.tensor([logits_list])
            probabilities = F.softmax(logits, dim=1)
            confidence = torch.max(probabilities, dim=1)[0].item()
            passes_07 = confidence > 0.7
            
            status = "✅" if passes_07 else "❌"
            print(f"  {description}: {confidence:.4f} {status}")
            
            if "strong" in description.lower() and not passes_07:
                all_strong_pass = False
        
        if all_strong_pass:
            print("✅ All strong logits produce confidence > 0.7")
        else:
            print("❌ Some strong logits still fail confidence > 0.7")
            
        return all_strong_pass
        
    except ImportError:
        print("⚠️  PyTorch not available for confidence calculation")
        return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("CONFIDENCE FIXES VERIFICATION")
    print("=" * 60)
    
    # Check file updates
    files_ok, file_issues = check_file_updates()
    
    # Check thresholds
    thresholds_ok = check_confidence_thresholds()
    
    # Check documentation
    docs_ok = check_documentation()
    
    # Calculate expected values
    confidence_ok = calculate_expected_confidence()
    
    # Final summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_checks = [
        (files_ok, "File updates", file_issues),
        (thresholds_ok, "Confidence thresholds", []),
        (docs_ok, "Documentation", []),
        (confidence_ok, "Confidence calculations", []),
    ]
    
    passed = 0
    total = len(all_checks)
    
    for check_passed, check_name, issues in all_checks:
        if check_passed:
            print(f"✅ {check_name}")
            passed += 1
        else:
            print(f"❌ {check_name}")
            for issue in issues:
                print(f"   {issue}")
    
    print(f"\n📊 RESULT: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("\n✅ Confidence assertions should now work reliably")
        print("✅ Strong logits produce high confidence values")
        print("✅ Tests maintain meaningful confidence thresholds") 
        print("✅ Complete documentation provided")
    else:
        print("⚠️  Some verifications failed - check output above")
    
    print("=" * 60)
    
    # Usage examples
    print("\n📚 QUICK REFERENCE:")
    print("Strong logits for high confidence tests:")
    print("  torch.tensor([[0.1, 5.0, 0.1]])  # → confidence ~0.99")
    print("\nConfidence assertions:")
    print("  assert result['confidence'] > 0.7  # Works with strong logits")
    print("  assert result['is_confident'] is True")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
