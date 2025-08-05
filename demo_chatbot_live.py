#!/usr/bin/env python3
"""
🎬 LIVE DEMO - Chatbot Integration Showcase
==========================================

Umfassende Demo der vollständigen Chatbot-Integration mit allen Features
"""

import asyncio
import time
import os
from typing import Dict, List, Any
from datetime import datetime

def print_header(title: str):
    """Schöner Header für Demo-Sections"""
    print(f"\n{'='*60}")
    print(f"🎬 {title}")
    print(f"{'='*60}")

def print_step(step: str, description: str):
    """Demo-Schritt anzeigen"""
    print(f"\n🔸 SCHRITT: {step}")
    print(f"   {description}")
    print(f"   {'-'*50}")

def simulate_api_response(query: str, has_context: bool = True) -> Dict[str, Any]:
    """Simuliert eine Chatbot-Antwort für Demo-Zwecke"""
    
    responses = {
        "Was ist eine Berufsunfähigkeitsversicherung?": {
            "response": """Eine Berufsunfähigkeitsversicherung (BU) ist eine der wichtigsten Versicherungen für Berufstätige. Sie zahlt eine monatliche Rente, wenn Sie aufgrund von Krankheit oder Unfall Ihren Beruf nicht mehr zu mindestens 50% ausüben können.

**Wichtige Punkte:**
- Zahlt bereits bei 50% Berufsunfähigkeit
- Orientiert sich am zuletzt ausgeübten Beruf
- Keine Verweisung auf andere Tätigkeiten
- Leistung bis zum vereinbarten Rentenalter

Die BU-Versicherung ist besonders wichtig, da die gesetzliche Erwerbsminderungsrente nur einen Bruchteil des letzten Einkommens abdeckt.""",
            "tokens_used": 385,
            "response_time_ms": 1420,
            "sources": ["BU-Grundlagen.pdf", "Versicherungsbedingungen_2024.pdf"] if has_context else [],
            "context_used": has_context
        },
        
        "Wie hoch sollte die BU-Rente sein?": {
            "response": """Die Höhe der BU-Rente sollte idealerweise 70-80% Ihres aktuellen Nettoeinkommens betragen. Das reicht meist aus, um den Lebensstandard zu halten, da im Leistungsfall oft keine Arbeitskosten mehr anfallen.

**Empfohlene Rentenhöhe:**
- Angestellte: 70-80% des Nettoeinkommens  
- Selbstständige: 80-90% (höhere laufende Kosten)
- Berufseinsteiger: Dynamik einbauen für Einkommenssteigerungen

**Wichtig:** Die Versicherung prüft bei Antragstellung Ihre Einkommensverhältnisse. Eine zu hohe Rente wird oft nicht akzeptiert oder führt zu hohen Beiträgen.""",
            "tokens_used": 298,
            "response_time_ms": 1180,
            "sources": ["BU-Beratungsleitfaden.pdf", "Produktinformation.pdf"] if has_context else [],
            "context_used": has_context
        },
        
        "Welche Risiken sind ausgeschlossen?": {
            "response": """BU-Versicherungen haben verschiedene Risikoausschlüsse, die je nach Anbieter variieren können:

**Häufige Ausschlüsse:**
- Kriegsereignisse und Bürgerkrieg
- Vorsatzdelikte und Straftaten
- Selbstmord in den ersten 3 Jahren
- Kernenergie-Unfälle
- Berufskrankheiten aus vorvertraglicher Zeit

**Bedingte Ausschlüsse:**
- Psychische Erkrankungen (oft zeitlich begrenzt)
- Rücken-/Gelenkleiden (bei manchen Anbietern)
- Alkohol-/Drogenabhängigkeit

**Tipp:** Moderne Tarife haben oft deutlich weniger Ausschlüsse. Ein Vergleich verschiedener Anbieter lohnt sich!""",
            "tokens_used": 312,
            "response_time_ms": 1350,
            "sources": ["Versicherungsbedingungen_2024.pdf", "Risikoausschlüsse.pdf"] if has_context else [],
            "context_used": has_context
        }
    }
    
    return responses.get(query, {
        "response": f"Das ist eine interessante Frage zu: '{query}'. In einem echten System würde hier eine detaillierte Antwort basierend auf den hochgeladenen Dokumenten stehen.",
        "tokens_used": 150,
        "response_time_ms": 900,
        "sources": ["Diverse_Dokumente.pdf"] if has_context else [],
        "context_used": has_context
    })

def demo_configuration():
    """Demo der Chatbot-Konfiguration"""
    print_header("CHATBOT CONFIGURATION SHOWCASE")
    
    print("🔧 Verfügbare Konfigurationsoptionen:")
    
    config_options = {
        "OpenAI Settings": {
            "model": "gpt-4o-mini (kostengünstig) | gpt-4-turbo (premium)",
            "temperature": "0.0 (deterministisch) - 2.0 (kreativ)",
            "max_tokens": "100-4000 (Response-Länge)",
            "timeout": "10-60 Sekunden"
        },
        "RAG Configuration": {
            "enable_rag": "true/false - Dokumentenkontext verwenden",
            "max_context_chunks": "1-10 - Anzahl relevanter Textblöcke",
            "similarity_threshold": "0.5-0.9 - Mindest-Ähnlichkeit",
            "context_chunk_size": "500-2000 - Zeichen pro Chunk"
        },
        "Conversation Management": {
            "max_conversation_length": "5-20 - Anzahl gespeicherter Turns",
            "system_prompt": "Rollenbasierte Instruktionen",
            "conversation_context": "Automatische Kontext-Erhaltung"
        },
        "Performance Settings": {
            "request_timeout": "Timeout für API-Calls",
            "max_retries": "Wiederholungen bei Fehlern",
            "enable_streaming": "Streaming-Responses (zukünftig)"
        }
    }
    
    for category, options in config_options.items():
        print(f"\n📋 {category}:")
        for option, description in options.items():
            print(f"   • {option}: {description}")
    
    print(f"\n✨ Optimale Standard-Konfiguration:")
    print(f"   🎯 Model: gpt-4o-mini (Balance aus Kosten/Qualität)")
    print(f"   🧠 RAG: Aktiviert mit 5 Chunks")  
    print(f"   💬 Konversation: 10 Turns Historie")
    print(f"   ⚡ Performance: 30s Timeout, 3 Retries")

def demo_rag_system():
    """Demo des RAG-Systems"""
    print_header("RAG SYSTEM (RETRIEVAL AUGMENTED GENERATION)")
    
    print_step("1", "Benutzer stellt Frage")
    user_query = "Was ist eine Berufsunfähigkeitsversicherung?"
    print(f"   👤 User: '{user_query}'")
    
    print_step("2", "Vector Search in Pinecone")
    print("   🔍 Suche ähnliche Textblöcke...")
    time.sleep(0.5)
    
    mock_search_results = [
        {
            "score": 0.89,
            "source": "BU-Grundlagen.pdf", 
            "text": "Eine Berufsunfähigkeitsversicherung bietet finanziellen Schutz...", 
            "chunk_type": "paragraph"
        },
        {
            "score": 0.84,
            "source": "Versicherungsbedingungen_2024.pdf",
            "text": "Bei Berufsunfähigkeit wird eine monatliche Rente gezahlt...",
            "chunk_type": "definition"
        },
        {
            "score": 0.78,
            "source": "BU-Ratgeber.pdf",
            "text": "Die BU-Versicherung ist eine der wichtigsten Absicherungen...",
            "chunk_type": "introduction"
        }
    ]
    
    print("   ✅ 3 relevante Chunks gefunden:")
    for i, result in enumerate(mock_search_results, 1):
        print(f"      {i}. Score: {result['score']:.2f} | {result['source']}")
        print(f"         Preview: {result['text'][:60]}...")
    
    print_step("3", "Kontext-Zusammenstellung")
    print("   📄 Kombiniere Chunks zu strukturiertem Kontext...")
    
    combined_context = """[BU-Grundlagen.pdf | Relevanz: 0.89]
Eine Berufsunfähigkeitsversicherung bietet finanziellen Schutz bei Verlust der Arbeitsfähigkeit durch Krankheit oder Unfall. Sie zahlt eine monatliche Rente.

[Versicherungsbedingungen_2024.pdf | Relevanz: 0.84]  
Bei Berufsunfähigkeit wird eine monatliche Rente gezahlt, wenn mindestens 50% der beruflichen Tätigkeit nicht mehr ausgeübt werden können.

[BU-Ratgeber.pdf | Relevanz: 0.78]
Die BU-Versicherung ist eine der wichtigsten Absicherungen für Berufstätige, da die gesetzliche Erwerbsminderungsrente meist nicht ausreicht."""
    
    print(f"   📝 Kontext ({len(combined_context)} Zeichen):")
    print(f"   {combined_context[:200]}...")
    
    print_step("4", "LLM-Anfrage mit Kontext")
    print("   🤖 Sende Frage + Kontext an GPT-4...")
    time.sleep(1)
    
    llm_prompt = f"""Frage: {user_query}

Relevante Dokumente:
{combined_context}

Bitte beantworte die Frage basierend auf den bereitgestellten Dokumenten."""
    
    print(f"   📨 Prompt-Größe: {len(llm_prompt)} Zeichen")
    
    print_step("5", "Intelligente Antwort")
    response = simulate_api_response(user_query, has_context=True)
    
    print(f"   🎯 Response: {response['response'][:200]}...")
    print(f"   ⚡ Zeit: {response['response_time_ms']}ms")
    print(f"   🎫 Tokens: {response['tokens_used']}")
    print(f"   📚 Quellen: {', '.join(response['sources'])}")

def demo_conversation_flow():
    """Demo einer vollständigen Konversation"""
    print_header("CONVERSATION FLOW DEMONSTRATION")
    
    conversation = [
        {
            "user": "Was ist eine Berufsunfähigkeitsversicherung?",
            "context": "Initial question - needs comprehensive explanation"
        },
        {
            "user": "Wie hoch sollte die Rente sein?",
            "context": "Follow-up question - refers to previous topic"
        },
        {
            "user": "Welche Risiken sind ausgeschlossen?",
            "context": "Detailed follow-up - needs specific information"
        }
    ]
    
    chat_history = []
    
    for i, turn in enumerate(conversation, 1):
        print_step(f"Turn {i}", f"User fragt: '{turn['user']}'")
        
        # Simuliere Conversation Context
        if chat_history:
            print(f"   💭 Konversations-Kontext: {len(chat_history)} vorherige Turns")
            print(f"   🔗 Bezieht sich auf: {chat_history[-1]['topic']}")
        
        # Simuliere Response
        response = simulate_api_response(turn['user'])
        
        print(f"   🤖 Assistant antwortet ({response['tokens_used']} tokens):")
        print(f"   {response['response'][:150]}...")
        
        if response['context_used']:
            print(f"   📚 Verwendete Quellen: {', '.join(response['sources'])}")
        
        # Add to history
        chat_history.append({
            "user_message": turn['user'],
            "assistant_response": response['response'][:100] + "...",
            "topic": "BU-Versicherung",
            "tokens": response['tokens_used']
        })
        
        print(f"   ⏱️  Response-Zeit: {response['response_time_ms']}ms")
    
    # Conversation Summary
    total_tokens = sum(turn['tokens'] for turn in chat_history)
    print(f"\n📊 Konversations-Statistiken:")
    print(f"   • Turns: {len(chat_history)}")
    print(f"   • Gesamt-Tokens: {total_tokens}")
    print(f"   • Durchschn. Response-Zeit: {sum(simulate_api_response(t['user'])['response_time_ms'] for t in conversation) // len(conversation)}ms")
    print(f"   • RAG-Nutzung: 100% (alle Antworten mit Kontext)")

def demo_error_handling():
    """Demo des Error Handlings"""
    print_header("ERROR HANDLING & RELIABILITY")
    
    error_scenarios = [
        {
            "name": "OpenAI Rate Limit",
            "description": "API-Limit erreicht",
            "solution": "Exponential Backoff + Retry (3x)",
            "recovery_time": "2-8 Sekunden"
        },
        {
            "name": "Network Timeout", 
            "description": "Verbindung unterbrochen",
            "solution": "Request Retry mit verlängertem Timeout",
            "recovery_time": "5-10 Sekunden"
        },
        {
            "name": "Invalid API Key",
            "description": "Authentifizierung fehlgeschlagen", 
            "solution": "Graceful Error Message + Fallback",
            "recovery_time": "Sofort"
        },
        {
            "name": "Pinecone Unavailable",
            "description": "Vector DB nicht erreichbar",
            "solution": "RAG deaktivieren, nur LLM verwenden",
            "recovery_time": "Sofort"
        },
        {
            "name": "Context Too Large",
            "description": "Kontext überschreitet Token-Limit",
            "solution": "Automatische Kontext-Kürzung",
            "recovery_time": "Sofort"
        }
    ]
    
    print("🛡️  Implementierte Fehlerbehandlungsstrategien:")
    
    for i, scenario in enumerate(error_scenarios, 1):
        print(f"\n   {i}. {scenario['name']}")
        print(f"      Problem: {scenario['description']}")
        print(f"      Lösung: {scenario['solution']}")
        print(f"      Recovery: {scenario['recovery_time']}")
        
        # Simulate error handling
        print(f"      Status: ✅ Implementiert")
    
    print(f"\n🎯 Robustheit-Features:")
    print(f"   • Automatische Retry-Logic mit Exponential Backoff")
    print(f"   • Graceful Degradation bei Service-Ausfällen")
    print(f"   • Comprehensive Logging für Debugging")  
    print(f"   • User-friendly Error Messages")
    print(f"   • Fallback auf Basic LLM ohne RAG")

def demo_performance_monitoring():
    """Demo des Performance Monitorings"""
    print_header("PERFORMANCE MONITORING & ANALYTICS")
    
    # Simuliere Performance-Daten
    mock_stats = {
        "session_stats": {
            "total_queries": 15,
            "successful_responses": 14,
            "failed_responses": 1,
            "rag_retrievals": 12,
            "avg_response_time_ms": 1340,
            "total_tokens_used": 4250
        },
        "rag_performance": {
            "avg_search_time_ms": 180,
            "avg_chunks_found": 4.2,
            "avg_similarity_score": 0.78,
            "cache_hit_rate": 0.15
        },
        "conversation_analytics": {
            "avg_turns_per_session": 3.8,
            "most_common_topics": ["BU-Grundlagen", "Leistungen", "Ausschlüsse"],
            "user_satisfaction_estimated": 0.87
        },
        "cost_tracking": {
            "total_cost_usd": 0.042,
            "avg_cost_per_query": 0.0028,
            "tokens_per_dollar": 101190
        }
    }
    
    print("📊 Live Performance Dashboard:")
    
    print(f"\n🚀 Session Performance:")
    for metric, value in mock_stats["session_stats"].items():
        if "time" in metric:
            print(f"   • {metric}: {value}ms")
        elif "rate" in metric or "ratio" in metric:
            print(f"   • {metric}: {value:.1%}")
        else:
            print(f"   • {metric}: {value}")
    
    print(f"\n🔍 RAG System Performance:")
    for metric, value in mock_stats["rag_performance"].items():
        if "time" in metric:
            print(f"   • {metric}: {value}ms")
        elif "rate" in metric or "score" in metric:
            print(f"   • {metric}: {value:.2f}")
        else:
            print(f"   • {metric}: {value}")
    
    print(f"\n💬 Conversation Analytics:")
    for metric, value in mock_stats["conversation_analytics"].items():
        if isinstance(value, list):
            print(f"   • {metric}: {', '.join(value)}")
        elif isinstance(value, float):
            print(f"   • {metric}: {value:.2f}")
        else:
            print(f"   • {metric}: {value}")
    
    print(f"\n💰 Cost Tracking:")
    for metric, value in mock_stats["cost_tracking"].items():
        if "cost" in metric:
            print(f"   • {metric}: ${value:.4f}")
        else:
            print(f"   • {metric}: {value:,}")
    
    print(f"\n🎯 Performance Insights:")
    print(f"   • 93% Success Rate (14/15 queries)")
    print(f"   • Durchschn. 1.34s Response-Zeit")
    print(f"   • RAG in 80% der Fälle verwendet")
    print(f"   • $0.028 Kosten pro 10 Fragen")
    print(f"   • Hohe User Satisfaction (87%)")

async def demo_live_interaction():
    """Demo einer Live-Interaktion"""
    print_header("LIVE INTERACTION SIMULATION")
    
    print("🎭 Simuliere reale Chatbot-Interaktion...")
    print("   (In einem echten System würde hier der OpenAI API-Call stattfinden)")
    
    # Simuliere typing indicator
    print(f"\n👤 User: Was ist eine Berufsunfähigkeitsversicherung?")
    print(f"🤖 Chatbot: 🤔 Denke nach...")
    
    # Simuliere processing time
    for i in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    
    print(f"\n")
    
    # Simuliere response
    response = simulate_api_response("Was ist eine Berufsunfähigkeitsversicherung?")
    
    print(f"🤖 Chatbot:")
    # Simuliere streaming response
    words = response['response'].split()
    for i, word in enumerate(words[:30]):  # Erste 30 Wörter für Demo
        print(word, end=" ", flush=True)
        if i % 8 == 7:  # Neue Zeile alle 8 Wörter
            print()
        time.sleep(0.05)  # Typing effect
    
    print("...\n")
    
    print(f"📚 Quellen: {', '.join(response['sources'])}")
    print(f"⚡ {response['response_time_ms']}ms | 🎫 {response['tokens_used']} tokens")
    
    print(f"\n💡 In der echten Anwendung:")
    print(f"   • Rich Terminal UI mit Farben")
    print(f"   • Echtzeit-Streaming von OpenAI")
    print(f"   • Interactive Commands (/help, /stats, etc.)")
    print(f"   • Persistente Konversations-Historie")

def main():
    """Hauptfunktion für Live Demo"""
    print("🎬 BU-PROCESSOR CHATBOT LIVE DEMO")
    print("=" * 40)
    print("Umfassende Demonstration aller Features")
    
    demos = [
        ("Configuration Showcase", demo_configuration),
        ("RAG System Deep Dive", demo_rag_system),
        ("Conversation Flow", demo_conversation_flow),
        ("Error Handling", demo_error_handling),
        ("Performance Monitoring", demo_performance_monitoring),
        ("Live Interaction", lambda: asyncio.run(demo_live_interaction()))
    ]
    
    print(f"\n📋 Demo-Programm:")
    for i, (title, _) in enumerate(demos, 1):
        print(f"   {i}. {title}")
    
    print(f"\n🚀 Starte Full Demo...")
    input("Drücke Enter zum Fortfahren...")
    
    for title, demo_func in demos:
        try:
            demo_func()
            input(f"\n⏸️  Demo '{title}' abgeschlossen. Enter für nächste Demo...")
        except KeyboardInterrupt:
            print(f"\n🛑 Demo unterbrochen.")
            break
        except Exception as e:
            print(f"\n❌ Demo-Fehler in '{title}': {e}")
            input("Enter zum Fortfahren...")
    
    print(f"\n🎉 LIVE DEMO ABGESCHLOSSEN!")
    print(f"\n🚀 Nächste Schritte:")
    print(f"   1. Setup: python setup_chatbot.py")
    print(f"   2. Test: python test_chatbot.py")
    print(f"   3. Chat: python cli.py chat")
    print(f"   4. API: Verwende BUProcessorChatbot in eigenen Apps")

if __name__ == "__main__":
    main()
