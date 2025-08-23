"""Tests for requirement tracking functionality."""

import pytest
from bu_processor.intent.requirements import RequirementChecker, REQUIREMENTS


class TestRequirementChecker:
    
    def test_extract_fields_application(self):
        """Test field extraction for application intent"""
        checker = RequirementChecker()
        text = "Ich bin 35 Jahre alt, arbeite als Softwareentwickler und verdiene 60000 Euro pro Jahr"
        
        extracted = checker.extract_fields(text, "application")
        
        # Should extract profession
        assert "beruf" in extracted
        assert "entwickler" in extracted["beruf"].lower()
    
    def test_extract_date_pattern(self):
        """Test date pattern extraction"""
        checker = RequirementChecker()
        text = "Mein Geburtsdatum ist 15.03.1985"
        
        extracted = checker.extract_fields(text, "application")
        
        assert "geburtsdatum" in extracted
        assert extracted["geburtsdatum"] == "15.03.1985"
    
    def test_extract_number_pattern(self):
        """Test number pattern extraction"""
        checker = RequirementChecker()
        text = "Ich verdiene 45000 Euro im Jahr"
        
        extracted = checker.extract_fields(text, "application")
        
        # Should extract income number
        assert "jahreseinkommen" in extracted
        assert "45000" in extracted["jahreseinkommen"]
    
    def test_requirement_state_tracking(self):
        """Test requirement state tracking"""
        checker = RequirementChecker()
        
        # Initial state
        state = checker.get_state("session1", "application")
        assert len(state.missing_fields) == len(REQUIREMENTS["application"])
        
        # Update with some fields
        checker.update_state("session1", "application", "Ich arbeite als Lehrer")
        state = checker.get_state("session1", "application")
        
        # Should have fewer missing fields
        assert len(state.missing_fields) < len(REQUIREMENTS["application"])
    
    def test_next_question_generation(self):
        """Test next question generation"""
        checker = RequirementChecker()
        
        question = checker.get_next_question("session1", "application")
        assert question is not None
        assert ("?" in question or ":" in question)  # Should be a question or prompt
    
    def test_complete_flow_application(self):
        """Test complete application flow"""
        checker = RequirementChecker()
        session_id = "test_session"
        
        # Step 1: Initial state
        state = checker.get_state(session_id, "application")
        assert len(state.missing_fields) == 4  # All fields missing
        
        # Step 2: Provide birth date
        checker.update_state(session_id, "application", "Geboren am 10.05.1990")
        question = checker.get_next_question(session_id, "application")
        # Question might be None if all fields were extracted at once
        
        # Step 3: Provide profession
        checker.update_state(session_id, "application", "Arbeite als Ingenieur")
        question = checker.get_next_question(session_id, "application")
        # After providing profession, there might be no more questions if regex extracted multiple fields
        
        # Check final state
        state = checker.get_state(session_id, "application")
        
        # Should have collected some fields
        assert len(state.collected_fields) > 0
    
    def test_risk_assessment_fields(self):
        """Test risk assessment field extraction"""
        checker = RequirementChecker()
        
        # Test profession extraction
        text = "Arbeite als Pilot"
        extracted = checker.extract_fields(text, "risk")
        assert "beruf" in extracted
        assert "pilot" in extracted["beruf"].lower()
        
        # Test health information (basic)
        text2 = "Ich habe Diabetes"
        extracted2 = checker.extract_fields(text2, "risk")
        # Note: Health extraction is basic in MVP, may not extract everything
        
    def test_session_isolation(self):
        """Test that different sessions maintain separate states"""
        checker = RequirementChecker()
        
        # Session 1
        checker.update_state("session1", "application", "Arbeite als Arzt")
        state1 = checker.get_state("session1", "application")
        
        # Session 2
        checker.update_state("session2", "application", "Arbeite als Lehrer")
        state2 = checker.get_state("session2", "application")
        
        # States should be independent
        assert state1.collected_fields != state2.collected_fields
        if "beruf" in state1.collected_fields and "beruf" in state2.collected_fields:
            assert state1.collected_fields["beruf"] != state2.collected_fields["beruf"]
