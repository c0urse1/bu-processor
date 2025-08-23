"""Requirements management for intent-based flows."""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import structlog

logger = structlog.get_logger("intent.requirements")


class FieldType(str, Enum):
    DATE = "date"
    TEXT = "text"
    NUMBER = "number"
    CHOICE = "choice"


@dataclass
class RequiredField:
    name: str
    field_type: FieldType
    description: str
    validation_pattern: Optional[str] = None
    choices: Optional[List[str]] = None
    follow_up_question: str = ""


# Intent-specific required fields
REQUIREMENTS = {
    "application": [
        RequiredField(
            name="geburtsdatum",
            field_type=FieldType.DATE,
            description="Geburtsdatum",
            validation_pattern=r"\d{1,2}\.\d{1,2}\.\d{4}",
            follow_up_question="Bitte nennen Sie Ihr Geburtsdatum (TT.MM.JJJJ):"
        ),
        RequiredField(
            name="beruf",
            field_type=FieldType.TEXT,
            description="Beruf/Tätigkeit",
            follow_up_question="Welchen Beruf üben Sie aktuell aus?"
        ),
        RequiredField(
            name="jahreseinkommen",
            field_type=FieldType.NUMBER,
            description="Jahreseinkommen in Euro",
            validation_pattern=r"\d+\.?\d*",
            follow_up_question="Wie hoch ist Ihr jährliches Bruttoeinkommen?"
        ),
        RequiredField(
            name="versicherungssumme",
            field_type=FieldType.NUMBER,
            description="Gewünschte monatliche Rente",
            validation_pattern=r"\d+\.?\d*",
            follow_up_question="Welche monatliche BU-Rente wünschen Sie?"
        )
    ],
    "risk": [
        RequiredField(
            name="beruf",
            field_type=FieldType.TEXT,
            description="Beruf/Tätigkeit",
            follow_up_question="Welchen Beruf üben Sie aus?"
        ),
        RequiredField(
            name="gesundheitsangaben",
            field_type=FieldType.TEXT,
            description="Gesundheitszustand/Vorerkrankungen",
            follow_up_question="Haben Sie Vorerkrankungen oder Beschwerden?"
        ),
        RequiredField(
            name="hobbys_risiken",
            field_type=FieldType.TEXT,
            description="Risikoreiche Hobbys/Aktivitäten",
            follow_up_question="Üben Sie risikoreiche Hobbys oder Sportarten aus?"
        )
    ]
}


@dataclass
class RequirementState:
    """Tracks requirement fulfillment state"""
    intent: str
    collected_fields: Dict[str, str] = field(default_factory=dict)
    missing_fields: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if self.intent in REQUIREMENTS:
            all_required = {field.name for field in REQUIREMENTS[self.intent]}
            self.missing_fields = all_required - set(self.collected_fields.keys())


class RequirementChecker:
    """Checks and tracks requirement fulfillment"""
    
    def __init__(self):
        self.states: Dict[str, RequirementState] = {}
    
    def get_state(self, session_id: str, intent: str) -> RequirementState:
        """Get or create requirement state for session"""
        key = f"{session_id}_{intent}"
        if key not in self.states:
            self.states[key] = RequirementState(intent=intent)
        return self.states[key]
    
    def extract_fields(self, text: str, intent: str) -> Dict[str, str]:
        """Extract required fields from user text"""
        extracted = {}
        
        if intent not in REQUIREMENTS:
            return extracted
        
        text_lower = text.lower()
        
        for field in REQUIREMENTS[intent]:
            if field.validation_pattern:
                pattern_matches = re.findall(field.validation_pattern, text)
                if pattern_matches:
                    extracted[field.name] = pattern_matches[0]
            
            # Simple keyword extraction (enhance as needed)
            if field.name == "beruf" and ("beruf" in text_lower or "arbeite als" in text_lower):
                # Extract profession with basic NLP
                words = text.split()
                if "als" in text_lower:
                    try:
                        as_index = next(i for i, word in enumerate(words) if word.lower() == "als")
                        if as_index < len(words) - 1:
                            extracted[field.name] = " ".join(words[as_index+1:as_index+3])
                    except StopIteration:
                        pass
        
        return extracted
    
    def get_next_question(self, session_id: str, intent: str) -> Optional[str]:
        """Get next follow-up question for missing requirements"""
        state = self.get_state(session_id, intent)
        
        if not state.missing_fields:
            return None
        
        # Get first missing field
        missing_field_name = next(iter(state.missing_fields))
        
        for field in REQUIREMENTS.get(intent, []):
            if field.name == missing_field_name:
                return field.follow_up_question
        
        return None
    
    def update_state(self, session_id: str, intent: str, user_text: str) -> RequirementState:
        """Update requirement state with new user input"""
        state = self.get_state(session_id, intent)
        extracted = self.extract_fields(user_text, intent)
        
        # Update collected fields
        state.collected_fields.update(extracted)
        
        # Update missing fields
        if intent in REQUIREMENTS:
            all_required = {field.name for field in REQUIREMENTS[intent]}
            state.missing_fields = all_required - set(state.collected_fields.keys())
        
        logger.info("Requirements updated",
                   session_id=session_id,
                   intent=intent,
                   extracted=extracted,
                   missing=list(state.missing_fields))
        
        return state
