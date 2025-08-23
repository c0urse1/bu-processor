#!/usr/bin/env python3
"""
Intent Recognition Demo for BU Processor

This script demonstrates the new intent recognition and routing system
that classifies user input into:
- advice (Beratung/FAQ)
- application (Antragserfassung)  
- risk (Risikoprüfung)
- oos (Out-of-scope)

Run with: python demo_intent_system.py
"""

import asyncio
import sys
from pathlib import Path

# Add the bu_processor package to the path
sys.path.insert(0, str(Path(__file__).parent))

from bu_processor.cli_query_understanding import handle_user_input
from bu_processor.intent.router import route


def print_banner():
    """Print demo banner"""
    print("🤖 BU-PROCESSOR INTENT RECOGNITION DEMO")
    print("=" * 50)
    print("Dieses Demo zeigt das neue Intent-Routing System:")
    print("• advice    - Beratung und FAQ")
    print("• application - Antragserfassung")
    print("• risk      - Risikoprüfung")
    print("• oos       - Out-of-scope")
    print("=" * 50)


async def demo_basic_routing():
    """Demo basic intent classification"""
    print("\n📋 BASIC INTENT CLASSIFICATION")
    print("-" * 30)
    
    test_inputs = [
        "Was ist eine Berufsunfähigkeitsversicherung?",
        "Ich möchte eine BU-Versicherung beantragen",
        "Ist mein Beruf als Pilot riskant?", 
        "Wie ist das Wetter heute?",
        "Erkläre mir die Vorteile einer BU-Versicherung",
        "Antrag stellen für Versicherung",
        "Gesundheitsfragen zur Risikoprüfung",
        "Kochrezepte für heute Abend"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        intent = route(user_input)
        print(f"{i:2d}. '{user_input[:40]}...' → {intent}")


async def demo_full_pipeline():
    """Demo full processing pipeline with responses"""
    print("\n🔄 FULL PROCESSING PIPELINE")
    print("-" * 30)
    
    test_cases = [
        {
            "input": "Was kostet eine Berufsunfähigkeitsversicherung?",
            "expected_intent": "advice",
            "description": "FAQ/Beratungsanfrage"
        },
        {
            "input": "Ich bin 30 Jahre alt und möchte eine BU-Versicherung abschließen",
            "expected_intent": "application", 
            "description": "Antragseinleitung"
        },
        {
            "input": "Ich arbeite als Dachdecker - wie hoch ist mein Risiko?",
            "expected_intent": "risk",
            "description": "Risikoprüfung"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['description']}")
        print(f"   Input: '{case['input']}'")
        
        try:
            result = await handle_user_input(case['input'], f"demo_session_{i}")
            print(f"   Intent: {result['intent']} (erwartet: {case['expected_intent']})")
            print(f"   Response: {result['response'][:100]}...")
            
            if result['intent'] == case['expected_intent']:
                print("   ✅ Korrekt geroutet")
            else:
                print("   ⚠️  Anderes Routing (kann bei Fallback normal sein)")
                
        except Exception as e:
            print(f"   ❌ Fehler: {e}")


async def demo_application_flow():
    """Demo application flow with requirement tracking"""
    print("\n📝 APPLICATION FLOW DEMO")
    print("-" * 30)
    
    session_id = "app_demo_session"
    
    conversation = [
        "Ich möchte eine BU-Versicherung beantragen",
        "Ich bin am 15.03.1985 geboren",
        "Ich arbeite als Softwareentwickler", 
        "Mein Jahreseinkommen beträgt 65000 Euro",
        "Ich möchte eine monatliche Rente von 2500 Euro"
    ]
    
    print("Simulating application conversation:")
    
    for i, user_input in enumerate(conversation, 1):
        print(f"\n{i}. User: {user_input}")
        
        try:
            result = await handle_user_input(user_input, session_id)
            print(f"   Assistant: {result['response'][:150]}...")
            
            if 'collected_fields' in result:
                print(f"   Gesammelte Daten: {result['collected_fields']}")
            if 'missing_fields' in result:
                print(f"   Fehlende Daten: {result['missing_fields']}")
                
        except Exception as e:
            print(f"   ❌ Fehler: {e}")


async def demo_risk_flow():
    """Demo risk assessment flow"""
    print("\n🔍 RISK ASSESSMENT DEMO")
    print("-" * 30)
    
    session_id = "risk_demo_session"
    
    conversation = [
        "Welche Risiken hat mein Beruf für eine BU-Versicherung?",
        "Ich arbeite als Feuerwehrmann",
        "Ich hatte vor 2 Jahren einen Bandscheibenvorfall",
        "Ich klettere gerne in der Freizeit"
    ]
    
    print("Simulating risk assessment conversation:")
    
    for i, user_input in enumerate(conversation, 1):
        print(f"\n{i}. User: {user_input}")
        
        try:
            result = await handle_user_input(user_input, session_id)
            print(f"   Assistant: {result['response'][:150]}...")
            
            if 'risk_score' in result:
                print(f"   Risiko-Score: {result['risk_score']}")
            if 'risk_level' in result:
                print(f"   Risiko-Level: {result['risk_level']}")
                
        except Exception as e:
            print(f"   ❌ Fehler: {e}")


async def interactive_demo():
    """Interactive demo allowing user input"""
    print("\n💬 INTERACTIVE DEMO")
    print("-" * 30)
    print("Geben Sie Fragen ein (oder 'quit' zum Beenden):")
    
    session_id = "interactive_session"
    
    while True:
        try:
            user_input = input("\nSie: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Demo beendet. Auf Wiedersehen!")
                break
                
            if not user_input:
                continue
                
            result = await handle_user_input(user_input, session_id)
            print(f"Assistant ({result['intent']}): {result['response']}")
            
        except KeyboardInterrupt:
            print("\nDemo beendet.")
            break
        except Exception as e:
            print(f"Fehler: {e}")


async def main():
    """Main demo function"""
    print_banner()
    
    # Basic routing demo
    await demo_basic_routing()
    
    # Full pipeline demo
    await demo_full_pipeline()
    
    # Application flow demo
    await demo_application_flow()
    
    # Risk assessment demo  
    await demo_risk_flow()
    
    # Interactive demo (optional)
    print("\n" + "=" * 50)
    choice = input("Möchten Sie das interaktive Demo starten? (y/n): ").strip().lower()
    if choice in ['y', 'yes', 'ja', 'j']:
        await interactive_demo()
    else:
        print("\nDemo beendet. Das Intent-System ist einsatzbereit! 🚀")


if __name__ == "__main__":
    asyncio.run(main())
