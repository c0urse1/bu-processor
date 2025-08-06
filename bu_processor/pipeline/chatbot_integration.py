#!/usr/bin/env python3
"""
🤖 CHATBOT INTEGRATION - RAG SYSTEM MIT OPENAI API
=================================================

Vollständige Chatbot-Integration für das BU-Processor System mit:
- OpenAI GPT-4 Integration für intelligente Antworten
- RAG (Retrieval Augmented Generation) mit Pinecone
- Konversations-Management und Kontext-Erhaltung
- Multiple Chat-Modi (Interactive, Single-Query, Batch)
- Fallback-Strategien und Error Handling
- Rate Limiting und Token Management
- Template-basierte Prompt-Generierung
- Context Truncation für Token-Limits
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import structlog
import re
from collections import deque

# OpenAI Integration
try:
    import openai
    from openai import OpenAI
    import tiktoken
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI nicht installiert. Installiere mit: pip install openai tiktoken")

# Template Engine
try:
    from jinja2 import Template, Environment, BaseLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("⚠️  Jinja2 nicht installiert. Installiere mit: pip install jinja2")

# Bestehende Pipeline-Integration
try:
    from .enhanced_integrated_pipeline import EnhancedIntegratedPipeline, EnhancedProcessingStrategy
    from .pinecone_integration import PineconeConfig, PineconePipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print("⚠️  Pipeline-Module nicht gefunden")

# CLI und UI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Rich nicht installiert. Installiere mit: pip install rich")

# Logging Setup
logger = structlog.get_logger("chatbot_integration")
console = Console() if RICH_AVAILABLE else None

# =============================================================================
# RATE LIMITING SYSTEM
# =============================================================================

@dataclass
class RateLimitConfig:
    """Konfiguration für Rate Limiting"""
    requests_per_minute: int = 60
    tokens_per_minute: int = 40000
    requests_per_hour: int = 3000
    tokens_per_hour: int = 1000000

class RateLimiter:
    """Token- und Request-Rate-Limiter für OpenAI API"""

    def __init__(self, config: RateLimitConfig):
        self.config = config

        # Request tracking
        self.request_times = deque()
        self.hourly_request_times = deque()

        # Token tracking
        self.token_usage = deque()
        self.hourly_token_usage = deque()

        # Locks für thread safety
        self._lock = asyncio.Lock()

    async def wait_if_needed(self, estimated_tokens: int = 0) -> Tuple[bool, str]:
        """
        Wartet falls Rate Limit erreicht wird.
        Returns: (can_proceed, wait_reason)
        """
        async with self._lock:
            now = datetime.now()

            # Cleanup alte Einträge
            self._cleanup_old_entries(now)

            # Check minute limits
            if len(self.request_times) >= self.config.requests_per_minute:
                wait_time = 60 - (now - self.request_times[0]).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return await self.wait_if_needed(estimated_tokens)

            current_minute_tokens = sum(t[0] for t in self.token_usage)
            if current_minute_tokens + estimated_tokens > self.config.tokens_per_minute:
                wait_time = 60 - (now - self.token_usage[0][1]).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return await self.wait_if_needed(estimated_tokens)

            # Check hourly limits
            if len(self.hourly_request_times) >= self.config.requests_per_hour:
                wait_time = 3600 - (now - self.hourly_request_times[0]).total_seconds()
                if wait_time > 0:
                    return False, f"Hourly request limit erreicht. Warte {wait_time:.0f}s"

            current_hour_tokens = sum(tokens for tokens, _ in self.hourly_token_usage)
            if current_hour_tokens + estimated_tokens > self.config.tokens_per_hour:
                return False, f"Hourly token limit erreicht. Versuche später erneut."

            return True, ""

    def record_request(self, tokens_used: int):
        """Registriert eine API-Anfrage und Token-Verbrauch"""
        now = datetime.now()

        self.request_times.append(now)
        self.hourly_request_times.append(now)
        self.token_usage.append((tokens_used, now))
        self.hourly_token_usage.append((tokens_used, now))

        # Cleanup
        self._cleanup_old_entries(now)

    def _cleanup_old_entries(self, now: datetime):
        """Entfernt alte Einträge aus den Tracking-Queues"""
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)

        # Minute cleanup
        while self.request_times and self.request_times[0] < minute_ago:
            self.request_times.popleft()

        while self.token_usage and self.token_usage[0][1] < minute_ago:
            self.token_usage.popleft()

        # Hour cleanup
        while self.hourly_request_times and self.hourly_request_times[0] < hour_ago:
            self.hourly_request_times.popleft()

        while self.hourly_token_usage and self.hourly_token_usage[0][1] < hour_ago:
            self.hourly_token_usage.popleft()

    def get_status(self) -> Dict[str, Any]:
        """Gibt aktuellen Rate-Limit-Status zurück"""
        now = datetime.now()
        self._cleanup_old_entries(now)

        minute_tokens = sum(tokens for tokens, _ in self.token_usage)
        hour_tokens = sum(tokens for tokens, _ in self.hourly_token_usage)

        return {
            "requests_this_minute": len(self.request_times),
            "requests_this_hour": len(self.hourly_request_times),
            "tokens_this_minute": minute_tokens,
            "tokens_this_hour": hour_tokens,
            "limits": {
                "requests_per_minute": self.config.requests_per_minute,
                "tokens_per_minute": self.config.tokens_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "tokens_per_hour": self.config.tokens_per_hour
            }
        }

# =============================================================================
# TOKEN MANAGEMENT UND TRUNCATION
# =============================================================================

class TokenManager:
    """Manages token counting und context truncation"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

        # Token limits für verschiedene Modelle
        self.model_limits = {
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385
        }

        self.max_tokens = self.model_limits.get(model, 8192)

        # Reserve tokens für Response
        self.response_reserve = 2000
        self.max_input_tokens = self.max_tokens - self.response_reserve

        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Zählt Tokens für einen Text"""
        try:
            return len(self.encoding.encode(text))
        except:
            # Fallback: rough estimation
            return int(len(text.split()) * 1.3)

    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Zählt Tokens für eine Nachrichtenliste"""
        total_tokens = 0

        for message in messages:
            # Base tokens pro Message
            total_tokens += 4  # Every message has role, content, name (optional), function_call (optional)

            for key, value in message.items():
                total_tokens += self.count_tokens(str(value))
                if key == "name":  # Name field
                    total_tokens -= 1  # Special case

        total_tokens += 2  # Every reply is primed with assistant

        return total_tokens

    def truncate_context(self, context_chunks: List[Dict[str, Any]], max_tokens: int) -> str:
        """Truncates context to fit within token limit based on relevance score."""
        if not context_chunks:
            return ""

        # Sort chunks by score in descending order
        sorted_chunks = sorted(context_chunks, key=lambda x: x.get('score', 0.0), reverse=True)

        truncated_chunks = []
        total_tokens = 0
        
        for chunk in sorted_chunks:
            chunk_text = f"[{chunk['source']} | Relevanz: {chunk['score']:.2f}]\n{chunk['text']}"
            chunk_tokens = self.count_tokens(chunk_text)

            if total_tokens + chunk_tokens <= max_tokens - 50:  # Reserve for truncation message
                truncated_chunks.append(chunk_text)
                total_tokens += chunk_tokens
            else:
                break
        
        if not truncated_chunks and sorted_chunks:
            # Fallback: take the best chunk and truncate it
            best_chunk = sorted_chunks[0]
            best_chunk_text = f"[{best_chunk['source']} | Relevanz: {best_chunk['score']:.2f}]\n{best_chunk['text']}"
            current_tokens = self.count_tokens(best_chunk_text)
            chars_per_token = len(best_chunk_text) / current_tokens
            max_chars = int(max_tokens * chars_per_token * 0.9)  # Safety margin
            return best_chunk_text[:max_chars] + "\n\n[KONTEXT GEKÜRZT]"

        result = '\n\n'.join(truncated_chunks)

        if len(truncated_chunks) < len(sorted_chunks):
            result += "\n\n[KONTEXT GEKÜRZT - WEITERE DOKUMENTE VERFÜGBAR]"

        return result


    def optimize_messages_for_limits(self, messages: List[Dict[str, str]], max_context_tokens: int = None) -> List[Dict[str, str]]:
        """Optimiert Messages um unter Token-Limit zu bleiben"""
        if max_context_tokens is None:
            max_context_tokens = self.max_input_tokens

        # Start mit allen Messages
        optimized_messages = messages.copy()
        current_tokens = self.count_message_tokens(optimized_messages)

        if current_tokens <= max_context_tokens:
            return optimized_messages

        # Strategy 1: Truncate user context in last message (assuming context is there)
        if len(optimized_messages) > 1:
            last_message = optimized_messages[-1]
            if last_message.get("role") == "user" and "Relevante Dokumente:" in last_message.get("content", ""):
                 # This part will be handled by the new truncate_context logic in RAGContextManager
                 pass

        # Strategy 2: Remove older conversation history
        current_tokens = self.count_message_tokens(optimized_messages)
        while current_tokens > max_context_tokens and len(optimized_messages) > 2:
            # Keep system message and last user message, remove older history
            system_msg = optimized_messages[0]
            last_user_msg = optimized_messages[-1]

            # Remove messages from position 1 onwards, but keep last message
            if len(optimized_messages) > 3:
                optimized_messages = [system_msg] + optimized_messages[3:]
            else:
                optimized_messages = [system_msg, last_user_msg]
                break

            current_tokens = self.count_message_tokens(optimized_messages)

        # Strategy 3: Final truncation of user message if still too large
        current_tokens = self.count_message_tokens(optimized_messages)
        if current_tokens > max_context_tokens and len(optimized_messages) >= 2:
            last_message = optimized_messages[-1]
            if last_message.get("role") == "user":
                # Calculate how much to truncate
                excess_tokens = current_tokens - max_context_tokens + 100  # Safety margin
                current_content_tokens = self.count_tokens(last_message["content"])
                max_content_tokens = max(200, current_content_tokens - excess_tokens)

                # Simple character-based truncation as a final fallback
                chars_per_token = len(last_message["content"]) / current_content_tokens
                max_chars = int(max_content_tokens * chars_per_token * 0.9)
                optimized_messages[-1]["content"] = last_message["content"][:max_chars] + "\n\n[INHALT GEKÜRZT]"


        logger.info("Token optimization completed",
                       original_tokens=self.count_message_tokens(messages),
                       optimized_tokens=self.count_message_tokens(optimized_messages),
                       messages_count=len(optimized_messages))

        return optimized_messages

# =============================================================================
# TEMPLATE SYSTEM
# =============================================================================

class PromptTemplateManager:
    """Verwaltet Template-basierte Prompt-Generierung"""

    def __init__(self):
        if not JINJA2_AVAILABLE:
            logger.warning("Jinja2 nicht verfügbar, verwende einfache String-Templates")
            self.jinja_env = None
        else:
            self.jinja_env = Environment(loader=BaseLoader())

        # Standard Templates
        self.templates = {
            "system_prompt": """Du bist ein hilfsreicher Assistent für {{ domain }}. 
Du beantwortest Fragen basierend auf den bereitgestellten Dokumenten präzise und hilfreich.
{% if language %}Antworte auf {{ language }}, es sei denn anders gewünscht.{% endif %}
{% if special_instructions %}
Besondere Anweisungen: {{ special_instructions }}
{% endif %}
Wenn die Informationen nicht in den Dokumenten stehen, sage das ehrlich.""",

            "user_with_context": """Frage: {{ question }}
{% if context %}
Relevante Dokumente:
{{ context }}

Bitte beantworte die Frage basierend auf den bereitgestellten Dokumenten.{% endif %}""",

            "fallback_response": """Basierend auf den verfügbaren Dokumenten:

{{ top_chunk_text }}

Quelle: {{ source }}

*Hinweis: Dies ist eine automatische Antwort, da der AI-Service momentan nicht verfügbar ist.*""",

            "context_chunk": """[{{ source }} | Relevanz: {{ score }}]
{{ text }}""",

            "error_response": """❌ {{ error_type }}: {{ error_message }}

{% if suggestions %}
💡 Lösungsvorschläge:
{% for suggestion in suggestions %}
- {{ suggestion }}
{% endfor %}
{% endif %}"""
        }

    def render_template(self, template_name: str, **kwargs) -> str:
        """Rendert ein Template mit gegebenen Variablen"""

        template_text = self.templates.get(template_name, "")
        if not template_text:
            logger.error("Template nicht gefunden", template_name=template_name)
            return ""

        try:
            if self.jinja_env:
                template = self.jinja_env.from_string(template_text)
                return template.render(**kwargs)
            else:
                # Einfache String-Substitution als Fallback
                return self._simple_template_render(template_text, **kwargs)

        except Exception as e:
            logger.error("Template-Rendering fehlgeschlagen",
                        template_name=template_name,
                        error=str(e))
            return template_text  # Return unrendered template

    def _simple_template_render(self, template_text: str, **kwargs) -> str:
        """Einfaches Template-Rendering ohne Jinja2"""
        result = template_text

        # Simple variable substitution
        for key, value in kwargs.items():
            placeholder = "{{ " + key + " }}"
            if placeholder in result:
                result = result.replace(placeholder, str(value) if value is not None else "")

        # Remove conditional blocks (simplified)
        result = re.sub(r'{%.*?%}', '', result, flags=re.DOTALL)

        return result.strip()

    def add_template(self, name: str, template_text: str):
        """Fügt ein neues Template hinzu"""
        self.templates[name] = template_text
        logger.debug("Template hinzugefügt", name=name)

    def get_available_templates(self) -> List[str]:
        """Gibt verfügbare Template-Namen zurück"""
        return list(self.templates.keys())

# =============================================================================
# FALLBACK SYSTEM
# =============================================================================

class FallbackManager:
    """Manages fallback strategies wenn OpenAI API nicht verfügbar ist"""

    def __init__(self, template_manager: PromptTemplateManager):
        self.template_manager = template_manager

    def generate_fallback_response(self, context: Optional['RetrievedContext'], error_type: str = "api_error") -> str:
        """Generiert eine Fallback-Antwort basierend auf verfügbarem Kontext"""

        if not context or not context.chunks:
            return self.template_manager.render_template(
                "error_response",
                error_type="Keine Daten verfügbar",
                error_message="Keine relevanten Dokumente gefunden",
                suggestions=[
                    "Formuliere deine Frage anders",
                    "Überprüfe die Rechtschreibung",
                    "Stelle eine allgemeinere Frage"
                ]
            )

        # Verwende den besten Chunk als Basis
        top_chunk = context.chunks[0]

        return self.template_manager.render_template(
            "fallback_response",
            top_chunk_text=top_chunk["text"][:800] + "..." if len(top_chunk["text"]) > 800 else top_chunk["text"],
            source=top_chunk["source"]
        )

    def generate_simple_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Generiert eine einfache Antwort basierend auf Heuristiken"""

        if not chunks:
            return "Keine relevanten Informationen zu dieser Frage gefunden."

        # Find best matching chunk
        best_chunk = max(chunks, key=lambda x: x.get("score", 0))

        # Simple keyword matching for common questions
        query_lower = query.lower()

        if any(word in query_lower for word in ["was ist", "definition", "bedeutung"]):
            response_start = "Basierend auf den Dokumenten: "
        elif any(word in query_lower for word in ["wie", "warum", "wann"]):
            response_start = "Laut den verfügbaren Informationen: "
        elif any(word in query_lower for word in ["kosten", "preis", "beitrag"]):
            response_start = "Zu den Kosten: "
        else:
            response_start = "Zu Ihrer Frage: "

        # Extract relevant text portion
        text = best_chunk["text"]
        if len(text) > 500:
            text = text[:500] + "..."

        return f"{response_start}\n\n{text}\n\nQuelle: {best_chunk['source']}"

# =============================================================================
# CHATBOT KONFIGURATION (ERWEITERT)
# =============================================================================

from pydantic import BaseModel, Field
from pinecone import Pinecone

@dataclass
class ChatbotConfig:
    """Erweiterte Konfiguration für das Chatbot-System"""

    # OpenAI Settings
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1500

    # Rate Limiting
    rate_limit_config: RateLimitConfig = field(default_factory=RateLimitConfig)

    # RAG Settings
    enable_rag: bool = True
    max_context_chunks: int = 5
    context_chunk_size: int = 800
    similarity_threshold: float = 0.7

    # Token Management
    max_context_tokens: int = 120000  # For gpt-4o-mini
    context_truncation_enabled: bool = True

    # Template Settings
    domain: str = "Berufsunfähigkeitsversicherungen (BU)"
    language: str = "Deutsch"
    special_instructions: str = ""

    # Conversation Settings
    max_conversation_length: int = 10

    # Pinecone Integration
    pinecone_index_name: str = "bu-processor-chat"
    pinecone_namespace: str = "documents"
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    pinecone_environment: str = field(default_factory=lambda: os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp-free"))

    # Fallback Settings
    enable_fallback: bool = True
    fallback_on_error: bool = True
    fallback_on_timeout: bool = True

    # Performance Settings
    request_timeout: int = 30
    max_retries: int = 3
    enable_streaming: bool = False

    def validate(self) -> Tuple[bool, List[str]]:
        """Validiert die Chatbot-Konfiguration"""
        errors = []

        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY environment variable nicht gesetzt")

        if not OPENAI_AVAILABLE:
            errors.append("OpenAI Python package nicht installiert")

        if self.enable_rag and not PIPELINE_AVAILABLE:
            errors.append("Pipeline-Module für RAG nicht verfügbar")

        if self.temperature < 0 or self.temperature > 2:
            errors.append("Temperature muss zwischen 0 und 2 liegen")

        if self.max_tokens < 100 or self.max_tokens > 4000:
            errors.append("Max tokens muss zwischen 100 und 4000 liegen")

        # Validiere Pinecone Index Name
        if self.enable_rag and self.pinecone_index_name:
            index_errors = self._validate_pinecone_index_name()
            errors.extend(index_errors)

        return len(errors) == 0, errors

    def _validate_pinecone_index_name(self) -> List[str]:
        """Validiert Pinecone Index Name und Existenz"""
        errors = []

        # Validiere Index Name Format
        if not self.pinecone_index_name:
            errors.append("Pinecone Index Name darf nicht leer sein")
            return errors

        # Pinecone Index Name Regeln
        if not re.match(r'^[a-z0-9-]+$', self.pinecone_index_name):
            errors.append("Pinecone Index Name darf nur Kleinbuchstaben, Zahlen und Bindestriche enthalten")

        if len(self.pinecone_index_name) < 1 or len(self.pinecone_index_name) > 45:
            errors.append("Pinecone Index Name muss zwischen 1 und 45 Zeichen lang sein")

        if self.pinecone_index_name.startswith('-') or self.pinecone_index_name.endswith('-'):
            errors.append("Pinecone Index Name darf nicht mit Bindestrich beginnen oder enden")

        # Prüfe Index-Existenz falls API Key verfügbar
        if self.pinecone_api_key and self.pinecone_environment:
            try:
                pc = Pinecone(api_key=self.pinecone_api_key)

                # Liste verfügbare Indizes
                try:
                    indexes = pc.list_indexes()
                    index_names = [idx.name for idx in indexes]

                    if self.pinecone_index_name not in index_names:
                        logger.warning(
                            "Pinecone Index nicht gefunden",
                            index_name=self.pinecone_index_name,
                            available_indexes=index_names
                        )
                        errors.append(
                            f"Pinecone Index '{self.pinecone_index_name}' existiert nicht. "
                            f"Verfügbare Indizes: {', '.join(index_names) if index_names else 'Keine'}"
                        )
                    else:
                        # Prüfe Index-Status
                        index_description = pc.describe_index(self.pinecone_index_name)
                        if index_description.status['ready'] != True:
                            errors.append(
                                f"Pinecone Index '{self.pinecone_index_name}' ist nicht bereit. "
                                f"Status: {index_description.status['state']}"
                            )

                        logger.info(
                            "Pinecone Index validiert",
                            index_name=self.pinecone_index_name,
                            dimension=index_description.dimension,
                            metric=index_description.metric,
                            pods=index_description.spec.pod.pods if hasattr(index_description.spec, 'pod') else 'serverless'
                        )

                except Exception as e:
                    logger.warning(
                        "Pinecone Index-Überprüfung fehlgeschlagen",
                        error=str(e),
                        index_name=self.pinecone_index_name
                    )
                    errors.append(f"Pinecone Index-Überprüfung fehlgeschlagen: {str(e)}")

            except Exception as e:
                logger.warning(
                    "Pinecone-Verbindung fehlgeschlagen",
                    error=str(e)
                )
                errors.append(f"Pinecone-Verbindung fehlgeschlagen: {str(e)}")

        elif self.enable_rag:
            errors.append("Pinecone API Key oder Environment nicht gesetzt aber RAG aktiviert")

        return errors

# =============================================================================
# RAG KONTEXT-MANAGER (ERWEITERT)
# =============================================================================

@dataclass
class RetrievedContext:
    """Container für retrieved context aus Pinecone"""
    query: str
    chunks: List[Dict[str, Any]]
    total_chunks: int
    search_time_ms: float
    context_text: str
    sources: List[str]
    truncated: bool = False
    original_token_count: int = 0
    final_token_count: int = 0

class RAGContextManager:
    """Manages context retrieval und preparation für RAG"""

    def __init__(self, config: ChatbotConfig, template_manager: PromptTemplateManager, token_manager: TokenManager):
        self.config = config
        self.template_manager = template_manager
        self.token_manager = token_manager
        self.pipeline = None

        if config.enable_rag and PIPELINE_AVAILABLE:
            try:
                self.pipeline = EnhancedIntegratedPipeline(
                    enable_pinecone=True,
                    pinecone_config={
                        "index_name": config.pinecone_index_name,
                        "embedding_model": "MULTILINGUAL_MINI"
                    }
                )
                logger.info("RAG Context Manager initialisiert",
                           index_name=config.pinecone_index_name)
            except Exception as e:
                logger.error("RAG Initialisierung fehlgeschlagen", error=str(e))
                self.pipeline = None

    async def retrieve_context(self, query: str, max_tokens: int = None) -> Optional[RetrievedContext]:
        """Ruft relevanten Kontext für eine Query ab"""

        if not self.pipeline:
            logger.warning("Kein RAG-System verfügbar, verwende leeren Kontext")
            return None

        start_time = time.time()

        try:
            search_results = self.pipeline.search_in_index(
                query=query,
                top_k=self.config.max_context_chunks,
                namespace=self.config.pinecone_namespace,
                similarity_threshold=self.config.similarity_threshold
            )

            search_time = (time.time() - start_time) * 1000

            if not search_results:
                logger.info("Keine relevanten Dokumente gefunden", query=query)
                return RetrievedContext(
                    query=query,
                    chunks=[],
                    total_chunks=0,
                    search_time_ms=search_time,
                    context_text="",
                    sources=[]
                )

            context_chunks = []
            sources = set()

            for result in search_results:
                metadata = result.get("metadata", {})
                text = metadata.get("text_preview", metadata.get("text", ""))
                source = metadata.get("source_file", "Unbekannte Quelle")

                if text and len(text.strip()) > 20:
                    context_chunks.append({
                        "text": text,
                        "source": source,
                        "score": result.get("score", 0.0),
                        "chunk_type": metadata.get("chunk_type", "unknown")
                    })
                    sources.add(source)

            # Generiere initialen context text
            context_text = self.token_manager.truncate_context(context_chunks, max_tokens or self.token_manager.max_input_tokens)
            original_token_count = sum(self.token_manager.count_tokens(f"[{c['source']} | Relevanz: {c['score']:.2f}]\n{c['text']}") for c in context_chunks)
            final_token_count = self.token_manager.count_tokens(context_text)
            truncated = original_token_count > final_token_count


            retrieved_context = RetrievedContext(
                query=query,
                chunks=context_chunks,
                total_chunks=len(context_chunks),
                search_time_ms=search_time,
                context_text=context_text,
                sources=list(sources),
                truncated=truncated,
                original_token_count=original_token_count,
                final_token_count=final_token_count
            )

            logger.info("Kontext erfolgreich abgerufen",
                       query=query,
                       chunks_found=len(context_chunks),
                       sources_count=len(sources),
                       search_time_ms=search_time,
                       original_tokens=original_token_count,
                       final_tokens=final_token_count,
                       truncated=truncated)

            return retrieved_context

        except Exception as e:
            logger.error("Kontext-Abruf fehlgeschlagen", query=query, error=str(e))
            return None

# =============================================================================
# KONVERSATIONS-MANAGER (UNVERÄNDERT)
# =============================================================================

@dataclass
class ConversationTurn:
    """Ein Turn in einer Konversation"""
    timestamp: datetime
    user_message: str
    assistant_response: str
    context_used: Optional[RetrievedContext] = None
    response_time_ms: float = 0.0
    tokens_used: int = 0

class ConversationManager:
    """Verwaltet Konversations-Historie und Kontext"""

    def __init__(self, config: ChatbotConfig, template_manager: PromptTemplateManager):
        self.config = config
        self.template_manager = template_manager
        self.conversation_history: List[ConversationTurn] = []
        self.conversation_id = f"chat_{int(time.time())}"

    def add_turn(self, turn: ConversationTurn):
        """Fügt einen neuen Turn zur Konversation hinzu"""
        self.conversation_history.append(turn)

        if len(self.conversation_history) > self.config.max_conversation_length:
            self.conversation_history = self.conversation_history[-self.config.max_conversation_length:]

        logger.debug("Conversation turn hinzugefügt",
                       conversation_id=self.conversation_id,
                       turn_count=len(self.conversation_history))

    def get_conversation_context(self) -> List[Dict[str, str]]:
        """Gibt Konversations-Kontext für OpenAI API zurück"""

        system_prompt = self.template_manager.render_template(
            "system_prompt",
            domain=self.config.domain,
            language=self.config.language,
            special_instructions=self.config.special_instructions
        )

        messages = [{"role": "system", "content": system_prompt}]

        for turn in self.conversation_history[-5:]:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_response})

        return messages

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Konversations-Statistiken zurück"""
        if not self.conversation_history:
            return {"total_turns": 0}

        total_tokens = sum(turn.tokens_used for turn in self.conversation_history)
        avg_response_time = sum(turn.response_time_ms for turn in self.conversation_history) / len(self.conversation_history)

        return {
            "conversation_id": self.conversation_id,
            "total_turns": len(self.conversation_history),
            "total_tokens_used": total_tokens,
            "avg_response_time_ms": avg_response_time,
            "context_retrievals": sum(1 for turn in self.conversation_history if turn.context_used),
            "start_time": self.conversation_history[0].timestamp.isoformat() if self.conversation_history else None,
            "last_interaction": self.conversation_history[-1].timestamp.isoformat() if self.conversation_history else None
        }

# =============================================================================
# HAUPTE CHATBOT-KLASSE (ERWEITERT)
# =============================================================================

class BUProcessorChatbot:
    """Hauptklasse für den BU-Processor Chatbot mit allen Verbesserungen"""

    def __init__(self, config: Optional[ChatbotConfig] = None):
        self.config = config or ChatbotConfig()

        # Validiere Konfiguration
        is_valid, errors = self.config.validate()
        if not is_valid:
            raise ValueError(f"Chatbot-Konfiguration ungültig: {'; '.join(errors)}")

        # Initialisiere Komponenten
        self.template_manager = PromptTemplateManager()
        self.token_manager = TokenManager(self.config.model)
        self.rate_limiter = RateLimiter(self.config.rate_limit_config)
        self.fallback_manager = FallbackManager(self.template_manager)

        # OpenAI Client
        if OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=self.config.openai_api_key)
        else:
            self.openai_client = None

        # RAG System
        self.rag_manager = RAGContextManager(self.config, self.template_manager, self.token_manager)

        # Konversations-Manager
        self.conversation_manager = ConversationManager(self.config, self.template_manager)

        # Performance-Statistiken
        self.stats = {
            "total_queries": 0,
            "successful_responses": 0,
            "fallback_responses": 0,
            "rag_retrievals": 0,
            "rate_limit_waits": 0,
            "context_truncations": 0,
            "avg_response_time_ms": 0.0,
            "total_tokens_used": 0
        }

        logger.info("BU-Processor Chatbot initialisiert",
                       model=self.config.model,
                       rag_enabled=self.config.enable_rag,
                       rate_limiting=True,
                       templating=JINJA2_AVAILABLE)

    async def chat(self, user_message: str, include_context: bool = True) -> Dict[str, Any]:
        """Hauptmethode für Chat-Interaktion mit allen Verbesserungen"""

        start_time = time.time()
        self.stats["total_queries"] += 1

        try:
            # 1. Estimate tokens for rate limiting
            estimated_tokens = self.token_manager.count_tokens(user_message) + 500  # Rough estimate

            # 2. Check rate limits
            can_proceed, wait_reason = await self.rate_limiter.wait_if_needed(estimated_tokens)

            if not can_proceed:
                logger.warning("Rate limit erreicht", reason=wait_reason)
                self.stats["rate_limit_waits"] += 1

                if self.config.fallback_on_timeout:
                    # Try fallback response
                    context = await self.rag_manager.retrieve_context(user_message) if include_context else None
                    fallback_response = self.fallback_manager.generate_fallback_response(context, "rate_limit")

                    return {
                        "response": fallback_response,
                        "fallback_used": True,
                        "fallback_reason": wait_reason,
                        "context_used": context is not None
                    }
                else:
                    return {
                        "response": f"⏱️ Rate Limit erreicht: {wait_reason}",
                        "error": "rate_limit",
                        "retry_after": wait_reason
                    }

            # 3. Retrieve context (RAG)
            context = None
            if include_context and self.config.enable_rag:
                max_context_tokens = self.token_manager.max_input_tokens // 2  # Reserve half for context
                context = await self.rag_manager.retrieve_context(user_message, max_context_tokens)
                if context:
                    self.stats["rag_retrievals"] += 1
                    if context.truncated:
                        self.stats["context_truncations"] += 1

            # 4. Prepare messages
            messages = self.conversation_manager.get_conversation_context()

            # 5. Add current user message with context using template
            if context and context.context_text:
                user_content = self.template_manager.render_template(
                    "user_with_context",
                    question=user_message,
                    context=context.context_text
                )
            else:
                user_content = user_message

            messages.append({"role": "user", "content": user_content})

            # 6. Optimize messages for token limits
            if self.config.context_truncation_enabled:
                messages = self.token_manager.optimize_messages_for_limits(messages)

            # 7. Try OpenAI API call
            response = await self._call_openai_api(messages)

            if "error" in response:
                # Fallback on API error
                if self.config.fallback_on_error:
                    fallback_response = self.fallback_manager.generate_fallback_response(context, response["error"])
                    self.stats["fallback_responses"] += 1

                    response_time = (time.time() - start_time) * 1000

                    turn = ConversationTurn(
                        timestamp=datetime.now(),
                        user_message=user_message,
                        assistant_response=fallback_response,
                        context_used=context,
                        response_time_ms=response_time,
                        tokens_used=0
                    )
                    self.conversation_manager.add_turn(turn)

                    return {
                        "response": fallback_response,
                        "fallback_used": True,
                        "fallback_reason": response["error"],
                        "response_time_ms": response_time,
                        "context_used": context is not None,
                        "sources": context.sources if context else []
                    }
                else:
                    return response

            assistant_response = response["content"]
            tokens_used = response.get("tokens_used", 0)

            # 8. Record rate limiting
            self.rate_limiter.record_request(tokens_used)

            # 9. Update statistics
            response_time = (time.time() - start_time) * 1000
            self.stats["successful_responses"] += 1
            self.stats["total_tokens_used"] += tokens_used
            if self.stats["successful_responses"] > 0:
                self.stats["avg_response_time_ms"] = (
                    (self.stats["avg_response_time_ms"] * (self.stats["successful_responses"] - 1) + response_time) /
                    self.stats["successful_responses"]
                )

            # 10. Save conversation turn
            turn = ConversationTurn(
                timestamp=datetime.now(),
                user_message=user_message,
                assistant_response=assistant_response,
                context_used=context,
                response_time_ms=response_time,
                tokens_used=tokens_used
            )
            self.conversation_manager.add_turn(turn)

            # 11. Return result
            result = {
                "response": assistant_response,
                "response_time_ms": response_time,
                "tokens_used": tokens_used,
                "context_used": context is not None,
                "sources": context.sources if context else [],
                "conversation_id": self.conversation_manager.conversation_id,
                "fallback_used": False
            }

            if context:
                result["context_details"] = {
                    "chunks_found": context.total_chunks,
                    "search_time_ms": context.search_time_ms,
                    "sources_count": len(context.sources),
                    "truncated": context.truncated,
                    "original_tokens": context.original_token_count,
                    "final_tokens": context.final_token_count
                }

            logger.info("Chat-Antwort erfolgreich generiert",
                       response_time_ms=response_time,
                       tokens_used=tokens_used,
                       context_used=context is not None,
                       context_truncated=context.truncated if context else False)

            return result

        except Exception as e:
            logger.error("Chat-Fehler", error=str(e), user_message=user_message[:100])

            # Try fallback on unexpected error
            if self.config.enable_fallback:
                try:
                    context = await self.rag_manager.retrieve_context(user_message) if include_context else None
                    fallback_response = self.fallback_manager.generate_fallback_response(context, "unexpected_error")
                    self.stats["fallback_responses"] += 1

                    return {
                        "response": fallback_response,
                        "fallback_used": True,
                        "fallback_reason": f"Unexpected error: {str(e)}",
                        "context_used": context is not None
                    }
                except:
                    pass

            return {
                "response": f"❌ Fehler bei der Antwort-Generierung: {str(e)}",
                "error": str(e),
                "context_used": False
            }

    async def _call_openai_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Ruft OpenAI API mit Retry-Logic und Fallback auf"""

        if not self.openai_client:
            return {"error": "OpenAI API nicht verfügbar"}

        for attempt in range(self.config.max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.request_timeout
                )

                return {
                    "content": response.choices[0].message.content,
                    "tokens_used": response.usage.total_tokens,
                    "model": response.model
                }

            except openai.RateLimitError as e:
                wait_time = 2 ** attempt
                logger.warning(f"Rate limit erreicht, warte {wait_time}s", attempt=attempt + 1)
                await asyncio.sleep(wait_time)

            except openai.APIError as e:
                logger.error("OpenAI API Fehler", error=str(e), attempt=attempt + 1)
                if attempt == self.config.max_retries - 1:
                    return {"error": f"OpenAI API Fehler: {str(e)}"}
                await asyncio.sleep(1)

            except Exception as e:
                logger.error("Unerwarteter OpenAI Fehler", error=str(e), attempt=attempt + 1)
                if attempt == self.config.max_retries - 1:
                    return {"error": f"Unerwarteter Fehler: {str(e)}"}
                await asyncio.sleep(1)

        # Wenn alle Retries fehlschlagen, Fallback-Logik anwenden
        logger.error("Maximale Anzahl an Wiederholungsversuchen für OpenAI API erreicht.")
        return {"error": "Maximale Anzahl Versuche erreicht"}


    def get_stats(self) -> Dict[str, Any]:
        """Gibt umfassende Statistiken zurück"""
        chatbot_stats = self.stats.copy()
        conversation_stats = self.conversation_manager.get_stats()
        rate_limit_status = self.rate_limiter.get_status()

        return {
            "chatbot_stats": chatbot_stats,
            "conversation_stats": conversation_stats,
            "rate_limit_status": rate_limit_status,
            "config": {
                "model": self.config.model,
                "rag_enabled": self.config.enable_rag,
                "max_context_chunks": self.config.max_context_chunks,
                "templating_enabled": JINJA2_AVAILABLE,
                "fallback_enabled": self.config.enable_fallback
            }
        }

    def reset_conversation(self):
        """Startet eine neue Konversation"""
        self.conversation_manager = ConversationManager(self.config, self.template_manager)
        logger.info("Neue Konversation gestartet")

# =============================================================================
# CLI INTEGRATION (ERWEITERT)
# =============================================================================

class ChatbotCLI:
    """CLI Interface für den erweiterten Chatbot"""

    def __init__(self):
        self.chatbot = None

    def interactive_chat(self):
        """Interactive chat session mit erweiterten Features"""
        if not RICH_AVAILABLE:
            print("❌ Rich nicht installiert. Verwende: pip install rich")
            return

        console.print(Panel.fit(
            "🤖 BU-Processor Chatbot (Enhanced)\n"
            "Features: RAG, Rate Limiting, Templates, Fallbacks\n"
            "Kommandos: /help, /stats, /reset, /config, /quit",
            title="Enhanced Chatbot gestartet",
            style="blue"
        ))

        try:
            config = ChatbotConfig()
            self.chatbot = BUProcessorChatbot(config)

            console.print("✅ Enhanced Chatbot erfolgreich initialisiert\n")

            while True:
                try:
                    user_input = Prompt.ask("[bold blue]Du[/bold blue]").strip()

                    if not user_input:
                        continue

                    if user_input.startswith('/'):
                        if self._handle_command(user_input):
                            break
                        continue

                    with console.status("[yellow]🤔 Verarbeite Anfrage...[/yellow]"):
                        result = asyncio.run(self.chatbot.chat(user_input))

                    if "error" in result:
                        console.print(f"[red]❌ Fehler: {result['error']}[/red]")
                    else:
                        response = result["response"]

                        console.print(f"[bold green]🤖 Assistant[/bold green]: {response}")

                        # Enhanced status information
                        status_parts = []
                        status_parts.append(f"⚡ {result['response_time_ms']:.0f}ms")
                        status_parts.append(f"🎫 {result['tokens_used']} tokens")

                        if result.get("fallback_used"):
                            status_parts.append(f"🔄 Fallback: {result.get('fallback_reason', 'unknown')}")

                        if result.get("context_used"):
                            context_details = result.get("context_details", {})
                            status_parts.append(f"📄 {context_details.get('chunks_found', 0)} chunks")

                            if context_details.get("truncated"):
                                status_parts.append("✂️ gekürzt")

                            sources = result.get("sources", [])
                            if sources:
                                console.print(f"[dim]📚 Quellen: {', '.join(sources[:3])}[/dim]")
                        else:
                            status_parts.append("💡 Ohne RAG")

                        console.print(f"[dim]{' | '.join(status_parts)}[/dim]")

                    console.print()

                except KeyboardInterrupt:
                    if Confirm.ask("\n🚪 Chat beenden?"):
                        break
                    console.print()

                except Exception as e:
                    console.print(f"[red]❌ Unerwarteter Fehler: {e}[/red]")
                    console.print()

        except Exception as e:
            console.print(f"[red]❌ Chatbot-Initialisierung fehlgeschlagen: {e}[/red]")

    def _handle_command(self, command: str) -> bool:
        """Handle special commands with new config command"""

        if command == '/quit' or command == '/q':
            console.print("👋 Auf Wiedersehen!")
            return True

        elif command == '/help' or command == '/h':
            console.print(Panel(
                "🤖 Enhanced Chatbot Kommandos:\n\n"
                "/help, /h     - Diese Hilfe anzeigen\n"
                "/stats, /s    - Erweiterte Statistiken anzeigen\n"
                "/config, /c   - Konfiguration anzeigen\n"
                "/reset, /r    - Konversation zurücksetzen\n"
                "/quit, /q     - Chat beenden\n\n"
                "🎯 Features: RAG, Rate Limiting, Templates, Fallbacks",
                title="Hilfe"
            ))

        elif command == '/stats' or command == '/s':
            if self.chatbot:
                stats = self.chatbot.get_stats()
                self._display_enhanced_stats(stats)
            else:
                console.print("[red]❌ Chatbot nicht initialisiert[/red]")

        elif command == '/config' or command == '/c':
            if self.chatbot:
                self._display_config()
            else:
                console.print("[red]❌ Chatbot nicht initialisiert[/red]")

        elif command == '/reset' or command == '/r':
            if self.chatbot:
                self.chatbot.reset_conversation()
                console.print("🔄 Konversation zurückgesetzt")
            else:
                console.print("[red]❌ Chatbot nicht initialisiert[/red]")

        else:
            console.print(f"[red]❓ Unbekanntes Kommando: {command}[/red]")

        return False

    def _display_enhanced_stats(self, stats: Dict[str, Any]):
        """Display enhanced chatbot statistics"""

        table = Table(title="🤖 Enhanced Chatbot Statistiken", show_header=True)
        table.add_column("Kategorie", style="cyan")
        table.add_column("Metrik", style="yellow")
        table.add_column("Wert", style="green")

        # Chatbot stats
        chatbot_stats = stats["chatbot_stats"]
        table.add_row("Performance", "Gesamte Anfragen", str(chatbot_stats["total_queries"]))
        table.add_row("", "Erfolgreiche Antworten", str(chatbot_stats["successful_responses"]))
        table.add_row("", "Fallback Antworten", str(chatbot_stats["fallback_responses"]))
        table.add_row("", "Ø Antwortzeit", f"{chatbot_stats['avg_response_time_ms']:.0f}ms")
        table.add_row("", "Tokens verwendet", str(chatbot_stats["total_tokens_used"]))

        table.add_row("RAG", "RAG-Abrufe", str(chatbot_stats["rag_retrievals"]))
        table.add_row("", "Context-Kürzungen", str(chatbot_stats["context_truncations"]))

        table.add_row("Rate Limiting", "Rate Limit Waits", str(chatbot_stats["rate_limit_waits"]))

        # Rate limit status
        rate_status = stats["rate_limit_status"]
        table.add_row("", "Requests/Min", f"{rate_status['requests_this_minute']}/{rate_status['limits']['requests_per_minute']}")
        table.add_row("", "Tokens/Min", f"{rate_status['tokens_this_minute']}/{rate_status['limits']['tokens_per_minute']}")

        console.print(table)

    def _display_config(self):
        """Display current configuration"""

        config = self.chatbot.config

        table = Table(title="⚙️ Chatbot Konfiguration", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Wert", style="green")

        table.add_row("Modell", config.model)
        table.add_row("Temperature", str(config.temperature))
        table.add_row("Max Tokens", str(config.max_tokens))
        table.add_row("RAG aktiviert", "✅" if config.enable_rag else "❌")
        table.add_row("Fallback aktiviert", "✅" if config.enable_fallback else "❌")
        table.add_row("Context Truncation", "✅" if config.context_truncation_enabled else "❌")
        table.add_row("Max Context Chunks", str(config.max_context_chunks))
        table.add_row("Domain", config.domain)
        table.add_row("Sprache", config.language)

        # Rate limiting
        rate_config = config.rate_limit_config
        table.add_row("Rate Limit (req/min)", str(rate_config.requests_per_minute))
        table.add_row("Rate Limit (tokens/min)", str(rate_config.tokens_per_minute))

        console.print(table)

# =============================================================================
# DEMO UND TESTING (ERWEITERT)
# =============================================================================

def demo_enhanced_chatbot():
    """Demo der erweiterten Chatbot-Integration"""

    print("🤖 ENHANCED CHATBOT INTEGRATION DEMO")
    print("=" * 50)

    print("🔧 Überprüfe Voraussetzungen...")

    issues = []
    if not OPENAI_AVAILABLE:
        issues.append("OpenAI/tiktoken package nicht installiert")

    if not os.getenv("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY nicht gesetzt")

    if issues:
        print("❌ Probleme gefunden:")
        for issue in issues:
            print(f"   - {issue}")
        return

    print("✅ Alle Voraussetzungen erfüllt")
    print(f"   - OpenAI: {'✅' if OPENAI_AVAILABLE else '❌'}")
    print(f"   - Jinja2: {'✅' if JINJA2_AVAILABLE else '❌'}")
    print(f"   - Pipeline: {'✅' if PIPELINE_AVAILABLE else '❌'}")
    print(f"   - Rich: {'✅' if RICH_AVAILABLE else '❌'}")

    # Enhanced configuration
    config = ChatbotConfig(
        model="gpt-4o-mini",
        enable_rag=True,
        max_context_chunks=3,
        context_truncation_enabled=True,
        enable_fallback=True,
        domain="Berufsunfähigkeitsversicherungen (BU)",
        language="Deutsch"
    )

    async def test_enhanced_features():
        try:
            chatbot = BUProcessorChatbot(config)

            test_scenarios = [
                {
                    "name": "Standard RAG Query",
                    "query": "Was ist eine Berufsunfähigkeitsversicherung?",
                    "test_rag": True
                },
                {
                    "name": "Long Context Test",
                    "query": "Erkläre mir alle Details zu Wartezeiten, Karenzzeiten, Leistungsausschlüssen und Beitragszahlungen bei BU-Versicherungen",
                    "test_truncation": True
                },
                {
                    "name": "Rate Limiting Test",
                    "query": "Welche Kosten entstehen?",
                    "test_rate_limit": True
                }
            ]

            for i, scenario in enumerate(test_scenarios, 1):
                print(f"\n🧪 Test {i}: {scenario['name']}")
                print(f"   Query: {scenario['query']}")

                result = await chatbot.chat(scenario['query'])

                if "error" in result:
                    print(f"   ❌ {result['error']}")
                else:
                    response_preview = result["response"][:100] + "..." if len(result["response"]) > 100 else result["response"]
                    print(f"   ✅ Antwort: {response_preview}")

                    # Enhanced metrics
                    print(f"   📊 Metriken:")
                    print(f"      ⚡ Zeit: {result['response_time_ms']:.0f}ms")
                    print(f"      🎫 Tokens: {result['tokens_used']}")
                    print(f"      🔄 Fallback: {'Ja' if result.get('fallback_used') else 'Nein'}")

                    if result.get("context_used"):
                        context = result.get("context_details", {})
                        print(f"      📄 Chunks: {context.get('chunks_found', 0)}")
                        print(f"      ✂️ Gekürzt: {'Ja' if context.get('truncated') else 'Nein'}")
                        if context.get("truncated"):
                            print(f"         Original: {context.get('original_tokens', 0)} tokens")
                            print(f"         Final: {context.get('final_tokens', 0)} tokens")

            # Final statistics
            stats = chatbot.get_stats()
            print(f"\n📊 ENHANCED DEMO STATISTIKEN:")
            print(f"   Anfragen: {stats['chatbot_stats']['total_queries']}")
            print(f"   Erfolgreich: {stats['chatbot_stats']['successful_responses']}")
            print(f"   Fallbacks: {stats['chatbot_stats']['fallback_responses']}")
            print(f"   RAG-Abrufe: {stats['chatbot_stats']['rag_retrievals']}")
            print(f"   Context-Kürzungen: {stats['chatbot_stats']['context_truncations']}")
            print(f"   Rate-Limit-Waits: {stats['chatbot_stats']['rate_limit_waits']}")

            rate_status = stats['rate_limit_status']
            print(f"   Rate Limit Status:")
            print(f"     Requests: {rate_status['requests_this_minute']}/{rate_status['limits']['requests_per_minute']} (min)")
            print(f"     Tokens: {rate_status['tokens_this_minute']}/{rate_status['limits']['tokens_per_minute']} (min)")

        except Exception as e:
            print(f"❌ Enhanced Demo fehlgeschlagen: {e}")

    asyncio.run(test_enhanced_features())

    print(f"\n🎉 ENHANCED CHATBOT DEMO ABGESCHLOSSEN!")

if __name__ == "__main__":
    demo_enhanced_chatbot()