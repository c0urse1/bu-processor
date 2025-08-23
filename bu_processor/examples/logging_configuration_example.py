#!/usr/bin/env python3
"""
Beispiel für die Verwendung der zentralisierten Logging-Konfiguration
==================================================================

Dieses Beispiel zeigt, wie die zentrale Logging-Konfiguration verwendet wird
und wie verschiedene Umgebungsvariablen das Verhalten beeinflussen.
"""

import os
import tempfile
from pathlib import Path

# Verschiedene Logging-Konfigurationen demonstrieren
def demo_console_logging():
    """Demonstriert Console-Logging (Standard)"""
    print("\n=== Console Logging Demo ===")
    
    # Standardkonfiguration (Console)
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["LOG_FORMAT"] = "console"
    
    # Import löst automatische Konfiguration aus
    from bu_processor.core.logging_setup import get_logger, get_logging_config
    
    logger = get_logger("example.console")
    config = get_logging_config()
    
    print(f"Aktuelle Konfiguration: {config}")
    
    logger.info("Console logging demo started", demo_type="console")
    logger.info("Processing document", document_id="DOC-123", pages=5, size_mb=2.1)
    logger.warning("Low disk space detected", available_gb=1.2, threshold_gb=2.0)
    logger.error("Processing failed", error_code="E001", retry_count=3)


def demo_json_logging():
    """Demonstriert JSON-Logging für strukturierte Logs"""
    print("\n=== JSON Logging Demo ===")
    
    # JSON-Format konfigurieren
    os.environ["LOG_FORMAT"] = "json"
    
    # Logging neu konfigurieren
    from bu_processor.core.logging_setup import reconfigure_logging, get_logger
    
    reconfigure_logging(log_format="json")
    logger = get_logger("example.json")
    
    logger.info("JSON logging demo started", demo_type="json")
    logger.info("User action", user_id="user123", action="login", ip="192.168.1.1")
    logger.info("API request", method="POST", endpoint="/classify", response_time_ms=250)


def demo_file_logging():
    """Demonstriert File-Logging mit Rotation"""
    print("\n=== File Logging Demo ===")
    
    # Temporäre Log-Datei
    with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as tmp_file:
        log_file = tmp_file.name
    
    # File-Logging konfigurieren
    os.environ["LOG_FILENAME"] = log_file
    os.environ["LOG_MAX_SIZE"] = "1"  # 1MB
    os.environ["LOG_BACKUP_COUNT"] = "3"
    
    from bu_processor.core.logging_setup import reconfigure_logging, get_logger
    
    reconfigure_logging(log_filename=log_file)
    logger = get_logger("example.file")
    
    logger.info("File logging demo started", log_file=log_file)
    logger.info("File will be rotated when size exceeds 1MB")
    
    # Prüfen ob Datei erstellt wurde
    if Path(log_file).exists():
        print(f"Log-Datei erstellt: {log_file}")
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Log-Inhalt:\n{content}")
    
    # Cleanup
    try:
        os.unlink(log_file)
    except:
        pass


def demo_bound_logger():
    """Demonstriert Bound Logger für kontextuelle Logs"""
    print("\n=== Bound Logger Demo ===")
    
    from bu_processor.core.logging_setup import get_bound_logger
    
    # Logger mit vordefiniertem Kontext
    request_logger = get_bound_logger("example.request", 
                                     request_id="REQ-789",
                                     user_id="user456")
    
    request_logger.info("Request processing started")
    request_logger.info("Validating input", field_count=5)
    request_logger.info("Classification completed", confidence=0.95, category="insurance")
    request_logger.info("Request processing completed", duration_ms=1250)


def demo_different_log_levels():
    """Demonstriert verschiedene Log-Level"""
    print("\n=== Log Levels Demo ===")
    
    from bu_processor.core.logging_setup import get_logger, reconfigure_logging
    
    # DEBUG-Level setzen
    reconfigure_logging(log_level="DEBUG")
    logger = get_logger("example.levels")
    
    logger.debug("Debug information", step="initialization", debug_data={"key": "value"})
    logger.info("Information message", status="processing")
    logger.warning("Warning message", warning_type="performance", threshold_exceeded=True)
    logger.error("Error occurred", error_type="validation", field="email")
    logger.critical("Critical system error", component="database", action_required="restart")


def main():
    """Hauptfunktion für alle Demos"""
    print("BU-Processor Centralized Logging Configuration Demo")
    print("=" * 60)
    
    try:
        demo_console_logging()
        demo_json_logging()
        demo_file_logging()
        demo_bound_logger()
        demo_different_log_levels()
        
        print("\n=== Demo abgeschlossen ===")
        print("Die zentrale Logging-Konfiguration bietet:")
        print("✅ Strukturierte Logs mit Schlüssel-Wert-Paaren")
        print("✅ Umgebungsvariablen-Konfiguration")
        print("✅ JSON- und Console-Output")
        print("✅ File-Logging mit Rotation")
        print("✅ Bound Logger für Kontext")
        print("✅ Automatische Konfiguration beim Package-Import")
        
    except Exception as e:
        print(f"Demo-Fehler: {e}")


if __name__ == "__main__":
    main()
