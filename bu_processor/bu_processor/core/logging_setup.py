"""
Zentrale Logging-Konfiguration für BU-Processor
==============================================

Einheitliches strukturiertes Logging mit Structlog für das gesamte Projekt.

Verwendung:
    from bu_processor.core.logging_setup import get_logger
    logger = get_logger(__name__)
    
    logger.info("processing started", document_id="doc123", file_count=5)
"""

import logging
import os
import sys
from typing import Optional

import structlog


def configure_logging() -> None:
    """
    Konfiguriert einheitliches strukturiertes Logging für das gesamte Projekt.
    
    Umgebungsvariablen:
        LOG_LEVEL: Logging-Level (DEBUG, INFO, WARNING, ERROR) - Standard: INFO
        LOG_FORMAT: Output-Format ("json" oder "console") - Standard: console
    """
    # Log-Level aus Umgebung oder Standard
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    
    # Output-Format aus Umgebung
    json_output = os.getenv("LOG_FORMAT", "console").lower() == "json"

    # Standard-Logger konfigurieren (für Kompatibilität)
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(message)s",  # Structlog übernimmt die Formatierung
        force=True  # Überschreibt vorhandene Konfiguration
    )

    # Gemeinsame Structlog-Prozessoren
    processors_common = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    
    # Format-spezifische Prozessoren
    if json_output:
        processors = processors_common + [
            structlog.processors.JSONRenderer()
        ]
    else:
        processors = processors_common + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    # Structlog konfigurieren
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Gibt einen konfigurierten strukturierten Logger zurück.
    
    Args:
        name: Logger-Name (typischerweise __name__)
        
    Returns:
        Strukturierter Logger mit Binding-Unterstützung
        
    Beispiel:
        logger = get_logger(__name__)
        logger.info("document processed", 
                   doc_id="123", 
                   pages=5, 
                   processing_time_ms=1250)
    """
    return structlog.get_logger(name)


def get_bound_logger(name: str, **context) -> structlog.stdlib.BoundLogger:
    """
    Gibt einen Logger mit vordefiniertem Kontext zurück.
    
    Args:
        name: Logger-Name
        **context: Kontext-Felder für alle Log-Nachrichten
        
    Returns:
        Logger mit gebundenem Kontext
        
    Beispiel:
        doc_logger = get_bound_logger(__name__, document_id="doc123")
        doc_logger.info("processing started")
        doc_logger.info("processing completed", pages=5)
    """
    return structlog.get_logger(name).bind(**context)


# Einmalige Konfiguration beim Import (falls noch nicht konfiguriert)
if not hasattr(structlog, '_configured'):
    configure_logging()
    structlog._configured = True
