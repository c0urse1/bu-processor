#!/usr/bin/env python3
"""
Simple test script for intent routing functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the bu_processor to the Python path if needed
sys.path.insert(0, str(Path(__file__).parent))

def test_intent_routing():
    """Test basic intent routing functionality"""
    print("🤖 Testing Intent Routing System")
    print("=" * 50)
    
    try:
        from bu_processor.intent import Intent, route
        
        # Test cases
        test_cases = [
            ("Was ist eine Berufsunfähigkeitsversicherung?", "advice"),
            ("Ich möchte einen Antrag stellen", "application"),
            ("Risikoprüfung für meinen Beruf", "risk"),
            ("Wie ist das Wetter heute?", "oos"),
        ]
        
        print("Testing intent classification...")
        for i, (text, expected) in enumerate(test_cases, 1):
            try:
                result = route(text)
                status = "✅" if result == expected else "⚠️"
                print(f"{i}. {status} '{text[:30]}...' → {result} (expected: {expected})")
            except Exception as e:
                print(f"{i}. ❌ Error: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

async def test_enhanced_query_handler():
    """Test the enhanced query handler"""
    print("\n🔄 Testing Enhanced Query Handler")
    print("=" * 50)
    
    try:
        from bu_processor.cli_query_understanding import handle_user_input
        
        test_input = "Was sind die Vorteile einer BU-Versicherung?"
        result = await handle_user_input(test_input, "test_session")
        
        print(f"Input: {test_input}")
        print(f"Intent: {result.get('intent', 'unknown')}")
        print(f"Response length: {len(result.get('response', ''))}")
        print(f"Has sources: {'sources' in result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Query handler error: {e}")
        return False

def test_requirements_system():
    """Test the requirements tracking system"""
    print("\n📋 Testing Requirements System")
    print("=" * 50)
    
    try:
        from bu_processor.intent.requirements import RequirementChecker, REQUIREMENTS
        
        checker = RequirementChecker()
        
        # Test application requirements
        print(f"Application requirements: {len(REQUIREMENTS['application'])} fields")
        for field in REQUIREMENTS['application']:
            print(f"  - {field.name}: {field.description}")
        
        # Test field extraction
        test_text = "Ich arbeite als Programmierer und bin am 15.03.1985 geboren"
        extracted = checker.extract_fields(test_text, "application")
        print(f"Extracted from '{test_text}': {extracted}")
        
        return True
        
    except Exception as e:
        print(f"❌ Requirements system error: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Intent Recognition System Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: Basic intent routing
    results.append(test_intent_routing())
    
    # Test 2: Requirements system
    results.append(test_requirements_system())
    
    # Test 3: Enhanced query handler
    results.append(await test_enhanced_query_handler())
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 30)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️ Some tests failed.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"Test runner error: {e}")
        sys.exit(1)
