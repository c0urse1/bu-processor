"""Intent schema definitions for BU Processor MVP."""

from enum import Enum
from typing import Dict, List
from pydantic import BaseModel


class Intent(str, Enum):
    """BU-specific intent categories for MVP"""
    ADVICE = "advice"           # Beratung/FAQ/Erklärung
    APPLICATION = "application" # Antragserfassung/Datenaufnahme
    RISK = "risk"              # Risikoprüfung/Voranfrage
    OOS = "oos"                # Out-of-scope/Escalation


class IntentConfig(BaseModel):
    """Configuration for intent classification"""
    confidence_threshold: float = 0.80
    default_intent: Intent = Intent.ADVICE
    fallback_enabled: bool = True


# Intent-specific keywords for initial classification
INTENT_KEYWORDS = {
    Intent.ADVICE: [
        "was ist", "wie funktioniert", "erklären", "bedeutung", "definition",
        "beratung", "hilfe", "verstehen", "frage", "info", "information",
        "erkläre mir", "was bedeutet", "wie geht", "warum", "wieso",
        "unterschied", "vergleich", "vorteile", "nachteile", "kosten"
    ],
    Intent.APPLICATION: [
        "antrag", "beantragen", "versicherung abschließen", "anmelden",
        "daten eingeben", "persönliche angaben", "vertrag", "abschluss",
        "möchte versicherung", "will beantragen", "antrag stellen",
        "anmeldung", "vertragsabschluss", "police", "abschließen"
    ],
    Intent.RISK: [
        "risikoprüfung", "gesundheit", "vorerkrankung", "beruf risiko",
        "gefährlich", "risikobewertung", "voranfrage", "prüfung",
        "risiko", "gesundheitsfragen", "vorerkrankungen", "berufskrankheit",
        "gefährdung", "risikofaktoren", "gesundheitscheck"
    ],
    Intent.OOS: [
        "wetter", "politik", "sport", "nachrichten", "kochrezept",
        "allgemein", "nicht versicherung", "auto", "kfz", "hausrat",
        "lebensversicherung", "krankenversicherung", "reiseversicherung",
        "tierversicherung", "rechtschutz"
    ]
}
