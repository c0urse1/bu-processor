from dataclasses import dataclass
from typing import Any, Dict, Tuple

@dataclass
class InputValidationConfig:
    enable_prompt_injection_detection: bool = True
    enable_openai_moderation: bool = False
    enable_pattern_filtering: bool = True
    enable_rate_limiting: bool = False
    enable_anomaly_detection: bool = False
    max_input_length: int = 4000
    suspicious_requests_per_minute: int | None = None
    suspicious_requests_per_hour: int | None = None
    prompt_injection_threshold: float | None = None
    anomaly_threshold: float | None = None
    openai_api_key: str | None = None
    log_suspicious_inputs: bool = True
    log_blocked_inputs: bool = True

class InputValidator:
    def __init__(self, cfg: InputValidationConfig):
        self.cfg = cfg
        self._stats: Dict[str, int] = {
            "validations": 0,
            "blocked": 0,
            "prompt_injections": 0,
            "anomalies": 0,
        }

    async def validate_input(self, text: str, user_id: str) -> Tuple[bool, str, Dict[str, Any]]:
        self._stats["validations"] += 1
        violations: list[str] = []
        sanitized = text.strip()

        # sehr einfache Platzhalter-Heuristiken
        if len(sanitized) > self.cfg.max_input_length:
            violations.append("length_limit")
        bad_patterns = ("@system", "ignore all previous instructions")
        if any(p in sanitized.lower() for p in bad_patterns):
            violations.append("prompt_injection")

        is_valid = len(violations) == 0
        if not is_valid:
            self._stats["blocked"] += 1
            if "prompt_injection" in violations:
                self._stats["prompt_injections"] += 1

        details = {"violations": violations}
        return is_valid, sanitized[: self.cfg.max_input_length], details

    def get_security_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

def sanitize_user_input(text: str) -> str:
    return text.strip()

def validate_chatbot_input(text: str) -> bool:
    return bool(text.strip())
