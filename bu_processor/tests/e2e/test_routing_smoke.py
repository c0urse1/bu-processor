"""End-to-end tests for intent routing and query handling."""

import pytest
import asyncio
from bu_processor.cli_query_understanding import handle_user_input


@pytest.mark.asyncio
class TestEndToEndRouting:
    
    async def test_advice_routing(self):
        """Test end-to-end advice routing"""
        response = await handle_user_input(
            "Was sind die Vorteile einer Berufsunfähigkeitsversicherung?",
            session_id="test_advice"
        )
        
        assert response["intent"] == "advice"
        assert "response" in response
        assert len(response["response"]) > 0
        assert "Entschuldigung" not in response["response"] or "error" not in response
    
    async def test_application_routing(self):
        """Test end-to-end application routing"""
        response = await handle_user_input(
            "Ich möchte eine BU-Versicherung beantragen",
            session_id="test_app"
        )
        
        # Should route to application and ask for requirements
        assert response["intent"] == "application"
        assert "collecting_requirements" in response.get("status", "") or "initiated" in response.get("status", "")
    
    async def test_risk_routing(self):
        """Test end-to-end risk assessment routing"""
        response = await handle_user_input(
            "Ich möchte eine Risikoprüfung für meinen Beruf machen",
            session_id="test_risk"
        )
        
        # Should route to risk assessment
        assert response["intent"] == "risk"
        assert "collecting_requirements" in response.get("status", "") or "initiated" in response.get("status", "")
    
    async def test_out_of_scope_routing(self):
        """Test out-of-scope routing"""
        response = await handle_user_input(
            "Wie ist das Wetter heute?",
            session_id="test_oos"
        )
        
        # Should route to out-of-scope or advice (fallback)
        assert response["intent"] in ["oos", "advice"]
        if response["intent"] == "oos":
            assert "MVP-Umfang" in response["response"]
    
    async def test_application_multi_turn(self):
        """Test multi-turn application conversation"""
        session_id = "test_app_multi"
        
        # Initial application request
        response1 = await handle_user_input(
            "Ich möchte eine BU-Versicherung beantragen",
            session_id=session_id
        )
        assert response1["intent"] == "application"
        
        # Provide birth date
        response2 = await handle_user_input(
            "Mein Geburtsdatum ist 15.03.1985",
            session_id=session_id
        )
        assert response2["intent"] == "application"
        
        # Should still be collecting or have progressed
        if "status" in response2:
            assert response2["status"] in ["collecting_requirements", "completed"]
    
    async def test_risk_multi_turn(self):
        """Test multi-turn risk assessment conversation"""
        session_id = "test_risk_multi"
        
        # Initial risk request
        response1 = await handle_user_input(
            "Risikoprüfung für meinen Beruf",
            session_id=session_id
        )
        assert response1["intent"] == "risk"
        
        # Provide profession
        response2 = await handle_user_input(
            "Ich arbeite als Pilot",
            session_id=session_id
        )
        assert response2["intent"] == "risk"
        
        # Should still be collecting or have progressed
        if "status" in response2:
            assert response2["status"] in ["collecting_requirements", "completed"]
    
    async def test_error_handling(self):
        """Test error handling for edge cases"""
        # Empty input
        response = await handle_user_input("", session_id="test_empty")
        assert "intent" in response
        assert "response" in response
        
        # Very long input
        long_text = "sehr langer text " * 100
        response = await handle_user_input(long_text, session_id="test_long")
        assert "intent" in response
        assert "response" in response
    
    async def test_session_isolation_e2e(self):
        """Test that different sessions maintain separate conversation states"""
        # Session 1: Start application
        response1a = await handle_user_input(
            "Antrag stellen",
            session_id="session1"
        )
        
        # Session 2: Start risk assessment  
        response2a = await handle_user_input(
            "Risikoprüfung",
            session_id="session2"
        )
        
        # Session 1: Continue application
        response1b = await handle_user_input(
            "Arbeite als Arzt",
            session_id="session1"
        )
        
        # Session 2: Continue risk assessment
        response2b = await handle_user_input(
            "Arbeite als Pilot",
            session_id="session2"
        )
        
        # Should maintain separate contexts
        assert response1a["intent"] in ["application", "advice"]
        assert response2a["intent"] in ["risk", "advice"]
        
        # If properly routed, contexts should be separate
        if response1b["intent"] == "application" and response2b["intent"] == "risk":
            # Sessions should have different collected data
            if "collected_fields" in response1b and "collected_fields" in response2b:
                assert response1b.get("collected_fields", {}) != response2b.get("collected_fields", {})


class TestSmokeTests:
    """Simple smoke tests that don't require async"""
    
    def test_import_intent_module(self):
        """Test that intent module can be imported"""
        from bu_processor.intent import Intent, route
        assert Intent.ADVICE == "advice"
        assert callable(route)
    
    def test_import_query_handler(self):
        """Test that enhanced query handler can be imported"""
        from bu_processor.cli_query_understanding import get_query_handler
        handler = get_query_handler()
        assert handler is not None
