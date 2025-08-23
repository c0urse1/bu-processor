#!/usr/bin/env python3
"""
Intent Routing Demo for BU Processor

This script demonstrates the new intent recognition and routing capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add the bu_processor to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from bu_processor.cli_query_understanding import handle_user_input


async def demo_intent_routing():
    """Interactive demo of intent routing functionality"""
    
    print("🤖 BU Processor Intent Routing Demo")
    print("=" * 50)
    print("Available intents:")
    print("• ADVICE - Beratung/FAQ/Erklärung")
    print("• APPLICATION - Antragserfassung")
    print("• RISK - Risikoprüfung")
    print("• OOS - Out-of-scope")
    print("\nType 'quit' to exit, 'demo' for automated demo\n")
    
    session_id = "demo_session"
    
    while True:
        try:
            user_input = input("👤 Sie: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Auf Wiedersehen! 👋")
                break
            
            if user_input.lower() == 'demo':
                await run_automated_demo()
                continue
            
            if not user_input:
                continue
            
            print("🔄 Verarbeitung...")
            result = await handle_user_input(user_input, session_id)
            
            print(f"🎯 Intent: {result['intent'].upper()}")
            if 'status' in result:
                print(f"📊 Status: {result['status']}")
            
            print(f"🤖 Antwort: {result['response']}")
            
            if 'sources' in result and result['sources']:
                print(f"📚 Quellen: {len(result['sources'])} gefunden")
            
            if 'collected_fields' in result and result['collected_fields']:
                print(f"📝 Gesammelte Daten: {result['collected_fields']}")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\nAuf Wiedersehen! 👋")
            break
        except Exception as e:
            print(f"❌ Fehler: {e}")


async def run_automated_demo():
    """Run automated demo with predefined inputs"""
    
    test_cases = [
        {
            "input": "Was ist eine Berufsunfähigkeitsversicherung?",
            "expected_intent": "advice",
            "description": "FAQ/Beratungsanfrage"
        },
        {
            "input": "Ich möchte eine BU-Versicherung beantragen",
            "expected_intent": "application", 
            "description": "Antragsstellung"
        },
        {
            "input": "Ist mein Beruf als Pilot riskant für eine BU-Versicherung?",
            "expected_intent": "risk",
            "description": "Risikoprüfung"
        },
        {
            "input": "Wie ist das Wetter heute?",
            "expected_intent": "oos",
            "description": "Out-of-scope Anfrage"
        }
    ]
    
    print("\n🎬 Automatische Demo")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"👤 Input: {test_case['input']}")
        
        try:
            result = await handle_user_input(test_case['input'], f"demo_{i}")
            
            intent = result['intent']
            print(f"🎯 Intent: {intent.upper()}")
            print(f"✅ Erwartet: {test_case['expected_intent'].upper()}")
            
            if intent == test_case['expected_intent']:
                print("✅ KORREKT")
            else:
                print("⚠️  ABWEICHUNG (kann bei Fallback-Logik normal sein)")
            
            print(f"🤖 Antwort: {result['response'][:100]}...")
            
        except Exception as e:
            print(f"❌ Fehler: {e}")
        
        print("-" * 30)
    
    print("\n✅ Demo abgeschlossen!")


async def test_multi_turn_application():
    """Test multi-turn application conversation"""
    
    print("\n🔄 Multi-Turn Application Test")
    print("=" * 50)
    
    session_id = "multi_turn_test"
    
    conversation = [
        "Ich möchte eine BU-Versicherung beantragen",
        "Mein Geburtsdatum ist 15.03.1985",
        "Ich arbeite als Softwareentwickler", 
        "Mein Jahreseinkommen beträgt 65000 Euro",
        "Ich möchte eine monatliche Rente von 2500 Euro"
    ]
    
    for i, message in enumerate(conversation, 1):
        print(f"\n{i}. 👤 Sie: {message}")
        
        try:
            result = await handle_user_input(message, session_id)
            
            print(f"🎯 Intent: {result['intent'].upper()}")
            if 'status' in result:
                print(f"📊 Status: {result['status']}")
            
            if 'collected_fields' in result and result['collected_fields']:
                print(f"📝 Gesammelte Daten: {result['collected_fields']}")
            
            if 'missing_fields' in result and result['missing_fields']:
                print(f"❓ Fehlende Daten: {result['missing_fields']}")
            
            print(f"🤖 Antwort: {result['response'][:200]}...")
            
        except Exception as e:
            print(f"❌ Fehler: {e}")
        
        print("-" * 30)


if __name__ == "__main__":
    try:
        asyncio.run(demo_intent_routing())
    except KeyboardInterrupt:
        print("\n\nDemo beendet.")
    except Exception as e:
        print(f"Demo-Fehler: {e}")
        sys.exit(1)
