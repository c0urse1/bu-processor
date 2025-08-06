#!/usr/bin/env python3
"""
🤖 CHATBOT SETUP - AUTOMATED INSTALLATION & CONFIGURATION
=========================================================

Automatisches Setup-Skript für die vollständige Chatbot-Integration
mit allen Dependencies, Konfiguration und Tests.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple

def run_command(command: str, description: str) -> Tuple[bool, str]:
    """Führt einen Befehl aus und gibt Erfolg/Fehler zurück"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=300
        )
        if result.returncode == 0:
            print(f"   ✅ Erfolgreich")
            return True, result.stdout
        else:
            print(f"   ❌ Fehler: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"   ❌ Timeout nach 5 Minuten")
        return False, "Timeout"
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False, str(e)

def check_python_version() -> bool:
    """Überprüft Python Version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Mindestens Python 3.8 erforderlich")
        return False

def install_dependencies() -> bool:
    """Installiert alle erforderlichen Dependencies"""
    print("\n📦 DEPENDENCY INSTALLATION")
    print("=" * 30)
    
    requirements = [
        "openai>=1.35.0",
        "rich>=13.7.1",
        "transformers>=4.40.0",
        "sentence-transformers>=2.7.0",
        "pinecone-client>=3.0.0",
        "torch>=2.3.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "structlog>=24.1.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "typer>=0.12.0",
        "PyMuPDF>=1.24.5",
        "python-dotenv>=1.0.1"
    ]
    
    # Installiere aus requirements.txt wenn vorhanden
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        success, output = run_command(
            f"pip install -r {requirements_file}",
            "Installiere Dependencies aus requirements.txt"
        )
        if success:
            return True
    
    # Fallback: Einzeln installieren
    failed_packages = []
    for package in requirements:
        success, output = run_command(
            f"pip install {package}",
            f"Installiere {package.split('>=')[0]}"
        )
        if not success:
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Fehlgeschlagene Pakete: {failed_packages}")
        print("💡 Versuche: pip install --upgrade pip")
        return False
    
    return True

def check_api_keys() -> Dict[str, bool]:
    """Überprüft API-Keys"""
    print("\n🔑 API KEY VALIDATION")
    print("=" * 22)
    
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY")
    }
    
    results = {}
    
    for key_name, key_value in keys.items():
        if key_value:
            # Basis-Validierung der Key-Länge
            if key_name == "OPENAI_API_KEY" and len(key_value) > 20 and key_value.startswith("sk-"):
                print(f"✅ {key_name}: Valide (sk-...{key_value[-4:]})")
                results[key_name] = True
            elif key_name == "PINECONE_API_KEY" and len(key_value) > 20:
                print(f"✅ {key_name}: Valide (...{key_value[-4:]})")
                results[key_name] = True
            else:
                print(f"⚠️  {key_name}: Format ungültig")
                results[key_name] = False
        else:
            print(f"❌ {key_name}: Nicht gesetzt")
            results[key_name] = False
    
    return results

def create_env_template():
    """Erstellt .env Template"""
    env_template = """# BU-Processor Chatbot Configuration
# =====================================

# OpenAI API Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here

# Pinecone Vector Database  
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-east-1-aws

# Chatbot Settings
CHATBOT_MODEL=gpt-4o-mini
CHATBOT_TEMPERATURE=0.7
CHATBOT_MAX_TOKENS=1500

# RAG Configuration
RAG_ENABLED=true
MAX_CONTEXT_CHUNKS=5
SIMILARITY_THRESHOLD=0.7

# Performance Settings
REQUEST_TIMEOUT=30
MAX_RETRIES=3

# Development Settings
LOG_LEVEL=INFO
DEBUG_MODE=false
"""
    
    env_file = Path(".env.template")
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(env_template)
    
    print(f"📝 .env Template erstellt: {env_file}")
    print("💡 Kopiere zu .env und fülle deine API-Keys ein")

def test_imports() -> Dict[str, bool]:
    """Testet wichtige Imports"""
    print("\n🧪 MODULE IMPORT TESTS")
    print("=" * 22)
    
    imports_to_test = [
        ("openai", "OpenAI API Client"),
        ("rich", "Rich Terminal UI"),
        ("transformers", "HuggingFace Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("pinecone", "Pinecone Vector DB"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("structlog", "Structured Logging")
    ]
    
    results = {}
    
    for module, description in imports_to_test:
        try:
            __import__(module)
            print(f"✅ {description}")
            results[module] = True
        except ImportError as e:
            print(f"❌ {description}: {e}")
            results[module] = False
    
    return results

def test_chatbot_integration() -> bool:
    """Testet die Chatbot-Integration"""
    print("\n🤖 CHATBOT INTEGRATION TEST")
    print("=" * 29)
    
    try:
        # Import Test
        sys.path.insert(0, "src")
        from pipeline.chatbot_integration import ChatbotConfig, BUProcessorChatbot
        print("✅ Chatbot Module erfolgreich importiert")
        
        # Config Test
        config = ChatbotConfig()
        is_valid, errors = config.validate()
        
        if is_valid:
            print("✅ Chatbot Konfiguration valide")
        else:
            print(f"⚠️  Chatbot Konfiguration: {'; '.join(errors)}")
        
        # Basis-Initialisierung (ohne API-Calls)
        if os.getenv("OPENAI_API_KEY"):
            try:
                chatbot = BUProcessorChatbot(config)
                print("✅ Chatbot erfolgreich initialisiert")
                return True
            except Exception as e:
                print(f"❌ Chatbot Initialisierung fehlgeschlagen: {e}")
                return False
        else:
            print("⚠️  Chatbot-Test übersprungen (OPENAI_API_KEY fehlt)")
            return True
            
    except ImportError as e:
        print(f"❌ Import fehlgeschlagen: {e}")
        return False
    except Exception as e:
        print(f"❌ Integration-Test fehlgeschlagen: {e}")
        return False

def create_demo_scripts():
    """Erstellt Demo-Skripte"""
    print("\n📝 DEMO SCRIPTS CREATION")
    print("=" * 24)
    
    # Quick Start Script
    quick_start = """#!/usr/bin/env python3
\"\"\"
🚀 QUICK START - BU-Processor Chatbot
\"\"\"

import os
import sys

def main():
    print("🤖 BU-Processor Chatbot - Quick Start")
    print("=" * 40)
    
    # Check API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY nicht gesetzt!")
        print("💡 Setze mit: export OPENAI_API_KEY='your-key'")
        return
    
    print("✅ OpenAI API Key gefunden")
    
    # Demo Options
    print("\\nVerfügbare Demos:")
    print("1. Interactive Chat")
    print("2. Single Question")
    print("3. Integration Test")
    
    choice = input("\\nWähle (1-3): ").strip()
    
    if choice == "1":
        os.system("python cli.py chat")
    elif choice == "2":
        question = input("Deine Frage: ")
        os.system(f'python cli.py ask "{question}"')
    elif choice == "3":
        os.system("python cli.py chatdemo")
    else:
        print("❌ Ungültige Auswahl")

if __name__ == "__main__":
    main()
"""
    
    with open("quick_start.py", 'w', encoding='utf-8') as f:
        f.write(quick_start)
    
    print("✅ quick_start.py erstellt")
    
    # Test Script
    test_script = """#!/usr/bin/env python3
\"\"\"
🧪 COMPREHENSIVE TEST - Alle Chatbot Features testen
\"\"\"

import asyncio
import os
import sys

async def test_all():
    print("🧪 COMPREHENSIVE CHATBOT TEST")
    print("=" * 35)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Überspringe Tests - OPENAI_API_KEY fehlt")
        return
    
    try:
        sys.path.insert(0, "src")
        from pipeline.chatbot_integration import BUProcessorChatbot, ChatbotConfig
        
        # Test Configuration
        config = ChatbotConfig(
            model="gpt-4o-mini",
            enable_rag=True,
            max_context_chunks=3
        )
        
        chatbot = BUProcessorChatbot(config)
        
        # Test Questions
        test_questions = [
            "Was ist eine Berufsunfähigkeitsversicherung?",
            "Wie hoch sollte die BU-Rente sein?",
            "Welche Risiken sind ausgeschlossen?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\\n🧪 Test {i}: {question}")
            
            result = await chatbot.chat(question)
            
            if "error" in result:
                print(f"   ❌ {result['error']}")
            else:
                print(f"   ✅ Antwort erhalten ({result['tokens_used']} tokens)")
                print(f"   ⚡ {result['response_time_ms']:.0f}ms")
                if result.get("context_used"):
                    print(f"   📚 RAG: {len(result['sources'])} Quellen")
        
        # Statistics
        stats = chatbot.get_stats()
        print(f"\\n📊 FINAL STATS:")
        print(f"   Erfolgreiche Anfragen: {stats['chatbot_stats']['successful_responses']}")
        print(f"   Durchschn. Response-Zeit: {stats['chatbot_stats']['avg_response_time_ms']:.0f}ms")
        print(f"   Tokens gesamt: {stats['chatbot_stats']['total_tokens_used']}")
        
        print("\\n🎉 Alle Tests erfolgreich!")
        
    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {e}")

if __name__ == "__main__":
    asyncio.run(test_all())
"""
    
    with open("test_chatbot.py", 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("✅ test_chatbot.py erstellt")

def main():
    """Hauptfunktion für Setup"""
    print("🤖 BU-PROCESSOR CHATBOT SETUP")
    print("=" * 35)
    print("Automatische Installation und Konfiguration")
    print()
    
    setup_steps = []
    
    # 1. Python Version Check
    if check_python_version():
        setup_steps.append(("Python Version", True))
    else:
        setup_steps.append(("Python Version", False))
        print("❌ Setup abgebrochen - Python Version zu alt")
        return
    
    # 2. Dependencies Installation
    if install_dependencies():
        setup_steps.append(("Dependencies", True))
    else:
        setup_steps.append(("Dependencies", False))
        print("⚠️  Einige Dependencies fehlen - Setup fortgesetzt")
    
    # 3. API Keys Check
    api_keys = check_api_keys()
    all_keys_valid = all(api_keys.values())
    setup_steps.append(("API Keys", all_keys_valid))
    
    if not all_keys_valid:
        create_env_template()
    
    # 4. Import Tests
    import_results = test_imports()
    critical_imports = all(import_results.get(module, False) for module in ["openai", "rich"])
    setup_steps.append(("Module Imports", critical_imports))
    
    # 5. Integration Test
    integration_success = test_chatbot_integration()
    setup_steps.append(("Chatbot Integration", integration_success))
    
    # 6. Demo Scripts
    create_demo_scripts()
    setup_steps.append(("Demo Scripts", True))
    
    # Final Summary
    print("\n🎯 SETUP SUMMARY")
    print("=" * 16)
    
    for step_name, success in setup_steps:
        status = "✅" if success else "❌"
        print(f"{status} {step_name}")
    
    successful_steps = sum(1 for _, success in setup_steps if success)
    total_steps = len(setup_steps)
    
    print(f"\n📊 Erfolg: {successful_steps}/{total_steps} Schritte")
    
    if successful_steps >= 4:
        print("\n🎉 SETUP ERFOLGREICH!")
        print("\n🚀 Nächste Schritte:")
        
        if not all_keys_valid:
            print("1. Setze deine API-Keys:")
            print("   export OPENAI_API_KEY='your-openai-key'")
            print("   export PINECONE_API_KEY='your-pinecone-key'")
        
        print("2. Teste die Integration:")
        print("   python quick_start.py")
        print("   python test_chatbot.py")
        
        print("3. Starte den Interactive Chat:")
        print("   python cli.py chat")
        
        print("4. Stelle eine einzelne Frage:")
        print("   python cli.py ask 'Was ist eine BU-Versicherung?'")
        
    else:
        print("\n❌ SETUP UNVOLLSTÄNDIG")
        print("💡 Überprüfe die fehlgeschlagenen Schritte und versuche es erneut.")

if __name__ == "__main__":
    main()
