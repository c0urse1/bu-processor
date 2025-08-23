"""Application intake flow for BU insurance applications."""

from typing import Dict, Any
import structlog

logger = structlog.get_logger("application.intake")


async def run_application_flow(user_text: str, session_id: str = "default") -> Dict[str, Any]:
    """
    Handle application intake flow (MVP stub).
    
    Args:
        user_text: User input text
        session_id: Session identifier for tracking
        
    Returns:
        Dict containing response and application state
    """
    logger.info("Starting application flow", session_id=session_id, text_preview=user_text[:50])
    
    # MVP: Simple acknowledgment and next steps
    response = """📋 **Antrag für Berufsunfähigkeitsversicherung**

Gerne helfe ich Ihnen bei Ihrem BU-Antrag! Für eine persönliche Beratung und den Antragsabschluss benötige ich einige Angaben von Ihnen.

**Nächste Schritte:**
1. Persönliche Daten erfassen
2. Berufliche Situation bewerten  
3. Gesundheitsfragen klären
4. Individuelle Beratung vereinbaren

Möchten Sie mit der Datenerfassung beginnen?"""
    
    return {
        "response": response,
        "intent": "application",
        "status": "initiated",
        "next_action": "collect_personal_data",
        "session_id": session_id
    }
