#!/usr/bin/env python3
"""
Enhanced CLI for query understanding with intent routing and multi-turn conversation.

Usage:
    python cli_query_understanding.py

This demonstrates:
- Intent-based routing (advice, application, risk, out-of-scope)
- Multi-turn chat conversation
- Query rewriting (heuristic/LLM)
- Query expansion (heuristic/LLM) 
- Multi-query retrieval with RRF fusion
- Requirement tracking for application and risk flows
"""

import asyncio
import structlog
from typing import Dict, Any, Optional

from .query.models import ChatTurn
from .factories import make_query_pipeline, make_hybrid_retriever
from .intent.router import route
from .intent.requirements import RequirementChecker
from .pipeline.chatbot_integration import BUProcessorChatbot
from .application.intake import run_application_flow
from .risk.engine import run_risk_assessment

logger = structlog.get_logger("cli.query_understanding")


class EnhancedQueryHandler:
    """Enhanced query handler with intent routing"""
    
    def __init__(self):
        self.chatbot = None
        self.requirement_checker = RequirementChecker()
        logger.info("Enhanced query handler initialized")
    
    def _get_chatbot(self):
        """Lazy initialization of chatbot"""
        if self.chatbot is None:
            try:
                self.chatbot = BUProcessorChatbot()
            except Exception as e:
                logger.warning("Chatbot initialization failed", error=str(e))
                self.chatbot = None
        return self.chatbot
    
    async def handle_user_input(
        self, 
        user_text: str, 
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Main entry point for user input processing with intent routing
        """
        try:
            # Step 1: Classify intent
            intent = route(user_text)
            
            logger.info("User input routed", 
                       intent=intent, 
                       session_id=session_id,
                       text_preview=user_text[:100])
            
            # Step 2: Route to appropriate handler
            if intent == "advice":
                return await self._handle_advice(user_text)
            
            elif intent == "application":
                return await self._handle_application(user_text, session_id)
            
            elif intent == "risk":
                return await self._handle_risk_assessment(user_text, session_id)
            
            else:  # oos
                return self._handle_out_of_scope(user_text)
        
        except Exception as e:
            logger.error("Query handling failed", error=str(e), user_text=user_text[:100])
            return {
                "response": "❌ Entschuldigung, bei der Verarbeitung Ihrer Anfrage ist ein Fehler aufgetreten.",
                "intent": "error",
                "error": str(e)
            }
    
    async def _handle_advice(self, user_text: str) -> Dict[str, Any]:
        """Handle advice/FAQ requests using existing RAG pipeline"""
        try:
            chatbot = self._get_chatbot()
            if chatbot:
                # Use existing chatbot with RAG
                result = await chatbot.chat(user_text, include_context=True)
                
                return {
                    "response": result["response"],
                    "intent": "advice",
                    "sources": result.get("sources", []),
                    "context_used": result.get("context_used", False),
                    "response_time_ms": result.get("response_time_ms", 0)
                }
            else:
                # Fallback to basic query understanding
                return await self._fallback_advice(user_text)
        
        except Exception as e:
            logger.error("Advice handling failed", error=str(e))
            return {
                "response": "Entschuldigung, ich kann Ihre Beratungsanfrage aktuell nicht bearbeiten. Bitte versuchen Sie es später erneut.",
                "intent": "advice",
                "error": str(e)
            }
    
    async def _fallback_advice(self, user_text: str) -> Dict[str, Any]:
        """Fallback advice handling using query pipeline"""
        try:
            # Use existing query pipeline as fallback
            chat_turns = [ChatTurn(role="user", content=user_text)]
            qp = make_query_pipeline()
            plan = qp.build_plan(chat_turns)
            
            # Try to retrieve relevant information
            try:
                retriever = make_hybrid_retriever()
                hits = qp.retrieve_union(plan, retriever, top_k_per_query=3, final_top_k=3)
                
                if hits:
                    response = f"Basierend auf Ihrer Frage zu '{plan.focused_query}' habe ich folgende Informationen gefunden:\n\n"
                    for i, hit in enumerate(hits, 1):
                        response += f"{i}. {hit.text[:200]}...\n\n"
                    return {
                        "response": response,
                        "intent": "advice",
                        "sources": [{"text": h.text, "score": h.score} for h in hits],
                        "context_used": True
                    }
                else:
                    return {
                        "response": "Entschuldigung, ich konnte keine spezifischen Informationen zu Ihrer Frage finden. Können Sie Ihre Frage präzisieren?",
                        "intent": "advice", 
                        "context_used": False
                    }
            except Exception:
                return {
                    "response": "Gerne helfe ich Ihnen bei Fragen zur Berufsunfähigkeitsversicherung. Können Sie Ihre Frage präzisieren?",
                    "intent": "advice",
                    "context_used": False
                }
        except Exception as e:
            logger.error("Fallback advice failed", error=str(e))
            return {
                "response": "Entschuldigung, ich kann Ihre Frage aktuell nicht bearbeiten.",
                "intent": "advice",
                "error": str(e)
            }
    
    async def _handle_application(self, user_text: str, session_id: str) -> Dict[str, Any]:
        """Handle application intake with requirement tracking"""
        try:
            # Update requirement state
            state = self.requirement_checker.update_state(session_id, "application", user_text)
            
            # Check if we need more information
            next_question = self.requirement_checker.get_next_question(session_id, "application")
            
            if next_question:
                return {
                    "response": next_question,
                    "intent": "application",
                    "status": "collecting_requirements",
                    "collected_fields": state.collected_fields,
                    "missing_fields": list(state.missing_fields),
                    "progress": len(state.collected_fields) / (len(state.collected_fields) + len(state.missing_fields)) if (len(state.collected_fields) + len(state.missing_fields)) > 0 else 0
                }
            else:
                # All requirements collected, proceed with application
                return await self._complete_application(state.collected_fields)
        
        except Exception as e:
            logger.error("Application handling failed", error=str(e))
            return {
                "response": "Bei der Antragsbearbeitung ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut.",
                "intent": "application",
                "error": str(e)
            }
    
    async def _handle_risk_assessment(self, user_text: str, session_id: str) -> Dict[str, Any]:
        """Handle risk assessment with requirement tracking"""
        try:
            # Update requirement state
            state = self.requirement_checker.update_state(session_id, "risk", user_text)
            
            # Check if we need more information
            next_question = self.requirement_checker.get_next_question(session_id, "risk")
            
            if next_question:
                return {
                    "response": next_question,
                    "intent": "risk",
                    "status": "collecting_requirements",
                    "collected_fields": state.collected_fields,
                    "missing_fields": list(state.missing_fields)
                }
            else:
                # All requirements collected, perform risk assessment
                return await self._perform_risk_assessment(state.collected_fields)
        
        except Exception as e:
            logger.error("Risk assessment handling failed", error=str(e))
            return {
                "response": "Bei der Risikoprüfung ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut.",
                "intent": "risk",
                "error": str(e)
            }
    
    def _handle_out_of_scope(self, user_text: str) -> Dict[str, Any]:
        """Handle out-of-scope requests"""
        return {
            "response": "Das liegt außerhalb des MVP-Umfangs. Ich kann dazu aktuell nichts Verlässliches sagen. Gerne helfe ich Ihnen bei Fragen zur Berufsunfähigkeitsversicherung!",
            "intent": "oos",
            "suggestion": "Versuchen Sie Fragen zu BU-Versicherungen, Anträgen oder Risikoprüfungen."
        }
    
    async def _complete_application(self, collected_fields: Dict[str, str]) -> Dict[str, Any]:
        """Complete application process (MVP stub)"""
        # MVP: Simple confirmation, later integrate with actual application system
        return {
            "response": f"""📋 **Antragsübersicht**

Ihre Angaben:
• Geburtsdatum: {collected_fields.get('geburtsdatum', 'N/A')}
• Beruf: {collected_fields.get('beruf', 'N/A')}
• Jahreseinkommen: {collected_fields.get('jahreseinkommen', 'N/A')} EUR
• Gewünschte BU-Rente: {collected_fields.get('versicherungssumme', 'N/A')} EUR

✅ Ihre Angaben wurden erfasst. Ein Berater wird sich mit Ihnen in Verbindung setzen.""",
            "intent": "application",
            "status": "completed",
            "collected_fields": collected_fields
        }
    
    async def _perform_risk_assessment(self, collected_fields: Dict[str, str]) -> Dict[str, Any]:
        """Perform risk assessment (MVP stub)"""
        # MVP: Simple rule-based assessment
        risk_score = self._calculate_risk_score(collected_fields)
        
        if risk_score < 3:
            risk_level = "NIEDRIG 🟢"
            message = "Basierend auf Ihren Angaben liegt ein niedriges Risiko vor."
        elif risk_score < 6:
            risk_level = "MITTEL 🟡"
            message = "Ihre Risikobewertung zeigt ein mittleres Risiko an."
        else:
            risk_level = "HOCH 🔴"
            message = "Die Bewertung zeigt ein erhöhtes Risiko an. Eine individuelle Prüfung ist erforderlich."
        
        return {
            "response": f"""🔍 **Risikoeinschätzung**

{message}

**Risikostufe:** {risk_level}
**Beruf:** {collected_fields.get('beruf', 'N/A')}
**Gesundheit:** {collected_fields.get('gesundheitsangaben', 'N/A')}
**Hobbys:** {collected_fields.get('hobbys_risiken', 'N/A')}

💡 Dies ist eine Vorabeinschätzung. Eine detaillierte Prüfung erfolgt im Antragsverfahren.""",
            "intent": "risk",
            "status": "completed",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "collected_fields": collected_fields
        }
    
    def _calculate_risk_score(self, fields: Dict[str, str]) -> int:
        """Simple rule-based risk scoring (MVP)"""
        score = 0
        
        beruf = fields.get('beruf', '').lower()
        gesundheit = fields.get('gesundheitsangaben', '').lower()
        hobbys = fields.get('hobbys_risiken', '').lower()
        
        # Job risk factors
        high_risk_jobs = ['pilot', 'bergsteiger', 'polizist', 'feuerwehr', 'bauarbeiter']
        if any(job in beruf for job in high_risk_jobs):
            score += 3
        
        # Health risk factors
        health_issues = ['diabetes', 'herz', 'krebs', 'depression', 'rücken']
        if any(issue in gesundheit for issue in health_issues):
            score += 2
        
        # Hobby risk factors
        extreme_sports = ['fallschirm', 'klettern', 'motor', 'boxen', 'ski']
        if any(sport in hobbys for sport in extreme_sports):
            score += 2
        
        return min(score, 10)  # Cap at 10


# Global handler instance
_handler_instance = None


def get_query_handler() -> EnhancedQueryHandler:
    """Get singleton query handler instance"""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = EnhancedQueryHandler()
    return _handler_instance


async def handle_user_input(user_text: str, session_id: str = "default") -> Dict[str, Any]:
    """Convenience function for handling user input"""
    handler = get_query_handler()
    return await handler.handle_user_input(user_text, session_id)


def run(chat_turns):
    """Legacy function for backward compatibility"""
    qp = make_query_pipeline()
    plan = qp.build_plan(chat_turns)
    print("Focused:", plan.focused_query)
    print("Expansions:", plan.expanded_queries)
    print("Trace:", plan.trace)
    print("All queries:", plan.all_queries)
    
    # Optional: retrieve across all queries (union + RRF)
    try:
        retriever = make_hybrid_retriever()
        hits = qp.retrieve_union(plan, retriever, top_k_per_query=5, final_top_k=5)
        print(f"\nRetrieved {len(hits)} results:")
        for i, h in enumerate(hits, 1):
            print(f"{i}. {h.score:.3f} [{h.metadata.get('section')}] {h.text[:90]}…")
    except Exception as e:
        print(f"Note: Retrieval skipped (no corpus): {e}")


async def demo_intent_routing():
    """Demo function showing intent routing capabilities"""
    handler = get_query_handler()
    
    test_inputs = [
        "Was ist eine Berufsunfähigkeitsversicherung?",
        "Ich möchte eine BU-Versicherung beantragen",
        "Ist mein Beruf als Programmierer riskant?",
        "Wie ist das Wetter heute?"
    ]
    
    print("🤖 Intent Routing Demo")
    print("=" * 50)
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n{i}. User: {user_input}")
        result = await handler.handle_user_input(user_input, f"demo_session_{i}")
        print(f"   Intent: {result['intent']}")
        print(f"   Response: {result['response'][:100]}...")


if __name__ == "__main__":
    # Legacy support
    chat = [
        ChatTurn(role="user", content="Hi, quick question about insurance."),
        ChatTurn(role="assistant", content="Sure, what do you need?"),
        ChatTurn(role="user", content="Which insurance covers financial loss from professional mistakes?"),
    ]
    run(chat)
    
    # New intent routing demo
    print("\n" + "="*60)
    asyncio.run(demo_intent_routing())
