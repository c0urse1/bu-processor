"""
Logging-Context-Helper für strukturierte Logs
===========================================

Bietet praktische Context-Manager für strukturierte Log-Kontexte.
"""

from contextlib import contextmanager
from typing import Any, Generator

import structlog


@contextmanager
def log_context(logger: structlog.stdlib.BoundLogger, **context: Any) -> Generator[structlog.stdlib.BoundLogger, None, None]:
    """
    Context-Manager für strukturierte Log-Kontexte.
    
    Args:
        logger: Basis-Logger
        **context: Kontext-Felder für alle Log-Nachrichten im Block
        
    Yields:
        Logger mit gebundenem Kontext
        
    Beispiel:
        from bu_processor.core.log_context import log_context
        
        with log_context(logger, document_id=doc_id, file_path=str(pdf_path)) as log:
            log.info("extraction started")
            # ... processing ...
            log.info("extraction completed", pages_processed=page_count)
    """
    bound_logger = logger.bind(**context)
    try:
        yield bound_logger
    finally:
        # Cleanup ist nicht nötig, da structlog bind() einen neuen Logger-Instance zurückgibt
        pass


@contextmanager
def timed_operation(logger: structlog.stdlib.BoundLogger, operation_name: str, **context: Any) -> Generator[structlog.stdlib.BoundLogger, None, None]:
    """
    Context-Manager für zeitgemessene Operationen mit strukturiertem Logging.
    
    Args:
        logger: Basis-Logger
        operation_name: Name der Operation für Logs
        **context: Zusätzliche Kontext-Felder
        
    Yields:
        Logger mit gebundenem Kontext inklusive operation_name
        
    Beispiel:
        with timed_operation(logger, "pdf_extraction", doc_id="123") as log:
            log.info("starting extraction")
            # ... processing ...
            # Automatisches End-Log mit Zeitdauer wird erstellt
    """
    import time
    
    start_time = time.time()
    operation_logger = logger.bind(operation=operation_name, **context)
    
    operation_logger.info("operation started")
    
    try:
        yield operation_logger
        
        duration_ms = int((time.time() - start_time) * 1000)
        operation_logger.info("operation completed", duration_ms=duration_ms)
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        operation_logger.error("operation failed", 
                             duration_ms=duration_ms,
                             error=str(e),
                             error_type=type(e).__name__)
        raise


def log_performance(logger: structlog.stdlib.BoundLogger, operation: str):
    """
    Decorator für Performance-Logging von Funktionen.
    
    Args:
        logger: Logger-Instance
        operation: Name der Operation
        
    Beispiel:
        @log_performance(logger, "document_classification")
        def classify_document(doc):
            # ... implementation ...
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with timed_operation(logger, operation, function=func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator
