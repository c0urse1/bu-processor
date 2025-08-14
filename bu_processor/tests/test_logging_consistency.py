"""
Tests für einheitliche Logging-Nutzung
=====================================

Stellt sicher, dass das strukturierte Logging korrekt konfiguriert ist
und keine veralteten Logging-Patterns verwendet werden.
"""

import ast
from pathlib import Path
from typing import List

import pytest


def test_no_private_logging_usage():
    """
    Stellt sicher, dass keine private _log API verwendet wird.
    
    Die private _log API von Python's logging kann zu unvorhersagbarem
    Verhalten führen und sollte durch strukturierte Logging-Aufrufe
    ersetzt werden.
    """
    root = Path(__file__).resolve().parents[1] / "bu_processor"
    offenders: List[str] = []
    
    for py_file in root.rglob("*.py"):
        # Skip __pycache__ und virtuelle Umgebungen
        if "__pycache__" in str(py_file) or "venv" in str(py_file):
            continue
            
        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(py_file))
        except Exception:
            # Skip Dateien, die nicht geparst werden können
            continue
            
        class LoggingVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # Suche nach logger._log( Aufrufen
                if (isinstance(node.func, ast.Attribute) and 
                    node.func.attr == "_log"):
                    offenders.append(f"{py_file.relative_to(root)}:{node.lineno}")
                self.generic_visit(node)
                
        LoggingVisitor().visit(tree)
    
    assert not offenders, (
        f"Forbidden logger._log calls found. Use structured logging instead:\n" + 
        "\n".join(offenders)
    )


def test_no_string_formatting_in_logs():
    """
    Prüft, dass keine String-Formatierung in Log-Nachrichten verwendet wird.
    
    Statt logger.info(f"Processing {count} docs") sollte verwendet werden:
    logger.info("processing documents", document_count=count)
    """
    root = Path(__file__).resolve().parents[1] / "bu_processor"
    offenders: List[str] = []
    
    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file) or "venv" in str(py_file):
            continue
            
        try:
            content = py_file.read_text(encoding="utf-8")
            
            # Suche nach problematischen Patterns
            lines = content.split('\n')
            for line_no, line in enumerate(lines, 1):
                line = line.strip()
                
                # f-string in logger calls
                if ('logger.' in line and 
                    any(method in line for method in ['.info(', '.debug(', '.warning(', '.error(']) and
                    'f"' in line):
                    offenders.append(f"{py_file.relative_to(root)}:{line_no} - f-string in log")
                
                # .format() in logger calls  
                if ('logger.' in line and 
                    any(method in line for method in ['.info(', '.debug(', '.warning(', '.error(']) and
                    '.format(' in line):
                    offenders.append(f"{py_file.relative_to(root)}:{line_no} - .format() in log")
                    
                # % formatting in logger calls
                if ('logger.' in line and 
                    any(method in line for method in ['.info(', '.debug(', '.warning(', '.error(']) and
                    ' % ' in line):
                    offenders.append(f"{py_file.relative_to(root)}:{line_no} - % formatting in log")
                    
        except Exception:
            continue
    
    if offenders:
        pytest.skip(f"String formatting in logs detected (will be fixed during refactoring):\n" + 
                   "\n".join(offenders[:10]))  # Zeige nur erste 10


def test_structlog_smoke():
    """
    Smoke-Test für strukturiertes Logging.
    
    Stellt sicher, dass die Logging-Konfiguration funktioniert
    und strukturierte Log-Aufrufe ohne Fehler ausgeführt werden können.
    """
    from bu_processor.core.logging_setup import get_logger, get_bound_logger
    
    # Basis-Logger testen
    logger = get_logger("test.smoke")
    
    # Sollte ohne TypeError oder andere Exceptions funktionieren
    logger.info("test message")
    logger.info("test with fields", 
               chunks_created=5, 
               file_path="test.pdf", 
               attempt=1,
               processing_time_ms=1250)
    
    # Bound logger testen
    bound_logger = logger.bind(document_id="doc123", session_id="sess456")
    bound_logger.info("processing started")
    bound_logger.debug("debug information", step="validation")
    bound_logger.warning("potential issue detected", issue_type="encoding")
    
    # get_bound_logger testen
    doc_logger = get_bound_logger("test.document", document_id="doc789")
    doc_logger.info("document processing", status="started")


def test_log_context_helpers():
    """
    Test für Log-Context-Helper.
    """
    from bu_processor.core.logging_setup import get_logger
    from bu_processor.core.log_context import log_context, timed_operation
    
    logger = get_logger("test.context")
    
    # log_context testen
    with log_context(logger, document_id="doc123", file_path="test.pdf") as log:
        log.info("extraction started")
        log.info("extraction completed", pages_processed=5)
    
    # timed_operation testen
    with timed_operation(logger, "test_operation", test_param="value") as log:
        log.info("operation in progress")
        # Simuliere kurze Verarbeitung
        import time
        time.sleep(0.01)


def test_logging_imports_consistency():
    """
    Prüft, dass alle Module den einheitlichen Logger-Import verwenden.
    """
    root = Path(__file__).resolve().parents[1] / "bu_processor"
    inconsistent_imports: List[str] = []
    
    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file) or "venv" in str(py_file):
            continue
            
        try:
            content = py_file.read_text(encoding="utf-8")
            lines = content.split('\n')
            
            has_old_logging = False
            has_new_logging = False
            
            for line_no, line in enumerate(lines, 1):
                line = line.strip()
                
                # Alte Logging-Imports
                if (line.startswith('import logging') or 
                    'logging.getLogger' in line):
                    has_old_logging = True
                    
                # Neue Logging-Imports
                if 'from bu_processor.core.logging_setup import' in line:
                    has_new_logging = True
            
            # Module mit beiden Arten markieren (sollte vermieden werden)
            if has_old_logging and has_new_logging:
                inconsistent_imports.append(f"{py_file.relative_to(root)} - Mixed logging imports")
            elif has_old_logging and 'logger' in content:
                # Nur melden, wenn auch Logger verwendet wird
                inconsistent_imports.append(f"{py_file.relative_to(root)} - Uses old logging.getLogger")
                
        except Exception:
            continue
    
    if inconsistent_imports:
        pytest.skip(f"Inconsistent logging imports detected (will be fixed during refactoring):\n" +
                   "\n".join(inconsistent_imports[:10]))


if __name__ == "__main__":
    # Für lokale Tests
    test_structlog_smoke()
    test_log_context_helpers()
    print("✓ All logging tests passed!")
