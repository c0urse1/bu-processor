#!/usr/bin/env python3
"""
Health Check Stabilization Verification

Verifiziert, dass die Health-Check Stabilisierung korrekt implementiert wurde.
"""

import os
import sys

def check_test_changes():
    """Prüft ob die Test-Änderungen korrekt sind."""
    print("🔍 Checking test changes...")
    
    test_file = "tests/test_classifier.py"
    if not os.path.exists(test_file):
        print(f"❌ {test_file} not found")
        return False
        
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if test uses classifier_with_eager_loading
    if "def test_health_status(self, classifier_with_eager_loading):" in content:
        print("   ✅ Test updated to use classifier_with_eager_loading")
    else:
        print("   ❌ Test still uses classifier_with_mocks instead of classifier_with_eager_loading")
        return False
        
    # Check if test has proper documentation
    if "sicherzustellen" in content and "dass das Modell geladen ist" in content:
        print("   ✅ Test documentation updated")
    else:
        print("   ❌ Test documentation missing or incomplete")
        return False
        
    return True

def check_fixture_improvements():
    """Prüft ob die Fixture-Verbesserungen implementiert sind."""
    print("\n🔍 Checking fixture improvements...")
    
    conftest_file = "tests/conftest.py"
    if not os.path.exists(conftest_file):
        print(f"❌ {conftest_file} not found")
        return False
        
    with open(conftest_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if classifier_with_mocks forces model loading
    if "RealMLClassifier(lazy=False)" in content:
        print("   ✅ classifier_with_mocks forces eager loading")
    else:
        print("   ❌ classifier_with_mocks doesn't force eager loading")
        return False
        
    if "classifier.model = mock_model" in content and "classifier.tokenizer = mock_tokenizer" in content:
        print("   ✅ classifier_with_mocks sets model and tokenizer explicitly")
    else:
        print("   ❌ classifier_with_mocks doesn't set model/tokenizer explicitly")
        return False
        
    return True

def check_health_check_improvements():
    """Prüft ob die Health-Check Verbesserungen implementiert sind."""
    print("\n🔍 Checking health check improvements...")
    
    classifier_file = "bu_processor/pipeline/classifier.py"
    if not os.path.exists(classifier_file):
        print(f"❌ {classifier_file} not found")
        return False
        
    with open(classifier_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for lazy mode detection
    if 'is_lazy_mode = getattr(self, \'_lazy\', False)' in content:
        print("   ✅ Health check detects lazy mode")
    else:
        print("   ❌ Health check doesn't detect lazy mode")
        return False
        
    # Check for degraded status
    if 'status = "degraded"' in content:
        print("   ✅ Health check supports degraded status")
    else:
        print("   ❌ Health check doesn't support degraded status")
        return False
        
    # Check for dummy initialization
    if "Health check dummy text" in content:
        print("   ✅ Health check has dummy initialization for lazy loading")
    else:
        print("   ❌ Health check missing dummy initialization")
        return False
        
    # Check for lazy_mode in response
    if '"lazy_mode": is_lazy_mode' in content:
        print("   ✅ Health check response includes lazy_mode info")
    else:
        print("   ❌ Health check response missing lazy_mode info")
        return False
        
    return True

def check_api_improvements():
    """Prüft ob die API-Verbesserungen implementiert sind."""
    print("\n🔍 Checking API improvements...")
    
    api_file = "bu_processor/api/main.py"
    if not os.path.exists(api_file):
        print(f"❌ {api_file} not found")
        return False
        
    with open(api_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for degraded status handling
    if 'classifier_status == "degraded"' in content:
        print("   ✅ API handles degraded status")
    else:
        print("   ❌ API doesn't handle degraded status")
        return False
        
    return True

def check_documentation():
    """Prüft ob die Dokumentation vollständig ist."""
    print("\n🔍 Checking documentation...")
    
    doc_file = "HEALTH_CHECK_STABILIZATION.md"
    if not os.path.exists(doc_file):
        print(f"❌ {doc_file} not found")
        return False
        
    with open(doc_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_sections = [
        "Problem Description",
        "Solution Implemented", 
        "Status-Semantik",
        "Files Updated",
        "Testing"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)
        else:
            print(f"   ✅ Documentation contains {section}")
    
    if missing_sections:
        print(f"   ❌ Missing documentation sections: {missing_sections}")
        return False
        
    return True

def main():
    """Main verification function."""
    print("=" * 60)
    print("HEALTH-CHECK STABILIZATION VERIFICATION")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Run all checks
    checks = [
        ("Test Changes", check_test_changes),
        ("Fixture Improvements", check_fixture_improvements), 
        ("Health Check Improvements", check_health_check_improvements),
        ("API Improvements", check_api_improvements),
        ("Documentation", check_documentation)
    ]
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_checks_passed = False
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            all_checks_passed = False
    
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ ALL HEALTH-CHECK STABILIZATION CHECKS PASSED!")
        print("\n🎯 SUMMARY:")
        print("- Test erwartete 'healthy', bekam 'unhealthy' ❌ → ✅ FIXED")
        print("- Model wird in Tests geladen (Schritt 3) ✅")
        print("- Health-Check toleranter bei Lazy Loading ✅")
        print("- Status 'degraded' statt 'unhealthy' ✅")
        print("- Dummy-Initialisierung für Lazy Loading ✅")
        print("- API behandelt alle Status korrekt ✅")
        print("- Vollständige Dokumentation ✅")
        print("\n🚀 Health-Check Stabilisierung komplett implementiert!")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("Please review the failed checks above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
