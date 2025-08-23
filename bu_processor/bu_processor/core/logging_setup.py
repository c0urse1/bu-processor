"""
Zentrale Logging-Konfiguration für BU-Processor
==============================================

Einheitliches strukturiertes Logging mit Structlog für das gesamte Projekt.
Bietet eine zentrale Konfiguration mit Umgebungsvariablen-Unterstützung.

Umgebungsvariablen:
    LOG_LEVEL: Logging-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL) - Standard: INFO
    LOG_FORMAT: Output-Format ("json", "console", "dev") - Standard: console
    LOG_FILENAME: Optional - Log-Datei für persistente Logs
    LOG_MAX_SIZE: Maximale Größe der Log-Datei in MB (Standard: 10)
    LOG_BACKUP_COUNT: Anzahl der Backup-Dateien (Standard: 5)

Verwendung:
    from bu_processor.core.logging_setup import get_logger
    logger = get_logger(__name__)
    
    logger.info("processing started", document_id="doc123", file_count=5)
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Try to import structlog with fallback
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None


class LoggingConfiguration:
    """Zentrale Logging-Konfiguration mit Environment-Unterstützung"""
    
    def __init__(self):
        self._configured = False
        self._log_level = self._get_log_level()
        self._log_format = self._get_log_format()
        self._log_filename = self._get_log_filename()
        self._max_size = self._get_max_size()
        self._backup_count = self._get_backup_count()
    
    def _get_log_level(self) -> int:
        """Log-Level aus Umgebungsvariable bestimmen"""
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        return getattr(logging, level_name, logging.INFO)
    
    def _get_log_format(self) -> str:
        """Log-Format aus Umgebungsvariable bestimmen"""
        return os.getenv("LOG_FORMAT", "console").lower()
    
    def _get_log_filename(self) -> Optional[str]:
        """Log-Dateiname aus Umgebungsvariable"""
        return os.getenv("LOG_FILENAME")
    
    def _get_max_size(self) -> int:
        """Maximale Log-Dateigröße in Bytes"""
        try:
            return int(os.getenv("LOG_MAX_SIZE", "10")) * 1024 * 1024  # MB to bytes
        except ValueError:
            return 10 * 1024 * 1024  # 10MB default
    
    def _get_backup_count(self) -> int:
        """Anzahl der Backup-Dateien"""
        try:
            return int(os.getenv("LOG_BACKUP_COUNT", "5"))
        except ValueError:
            return 5
    
    @property
    def is_configured(self) -> bool:
        """Prüft ob Logging bereits konfiguriert wurde"""
        return self._configured


# Globale Konfigurationsinstanz
_config = LoggingConfiguration()


def configure_logging() -> None:
    """
    Konfiguriert einheitliches strukturiertes Logging für das gesamte Projekt.
    
    Diese Funktion wird automatisch beim ersten Import aufgerufen und konfiguriert
    das Logging-System basierend auf Umgebungsvariablen.
    """
    global _config
    
    if _config.is_configured:
        return
    
    try:
        if STRUCTLOG_AVAILABLE:
            _configure_structlog()
        else:
            _configure_stdlib_logging()
        
        _config._configured = True
        
        # Log successful configuration
        if STRUCTLOG_AVAILABLE:
            logger = structlog.get_logger("core.logging_setup")
            logger.info("Logging system configured", 
                       structlog_available=True,
                       log_level=logging.getLevelName(_config._log_level),
                       log_format=_config._log_format,
                       log_file=_config._log_filename)
        else:
            logger = logging.getLogger("core.logging_setup")
            logger.info(f"Logging system configured [structlog_available=False, "
                       f"log_level={logging.getLevelName(_config._log_level)}, "
                       f"log_format={_config._log_format}]")
            
    except Exception as e:
        # Fallback to basic logging if configuration fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
        logger = logging.getLogger("core.logging_setup")
        logger.error(f"Failed to configure structured logging, using basic fallback: {e}")


def _configure_structlog() -> None:
    """Konfiguriert Structlog mit allen Features"""
    global _config
    
    # Standard-Logger für Kompatibilität konfigurieren
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Optional: File Handler hinzufügen
    if _config._log_filename:
        log_path = Path(_config._log_filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=_config._log_filename,
            maxBytes=_config._max_size,
            backupCount=_config._backup_count,
            encoding='utf-8'
        )
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=_config._log_level,
        handlers=handlers,
        format="%(message)s",  # Structlog übernimmt die Formatierung
        force=True
    )
    
    # Structlog-Prozessoren basierend auf Format
    processors_common = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    
    if _config._log_format == "json":
        processors = processors_common + [
            structlog.processors.JSONRenderer()
        ]
    elif _config._log_format == "dev":
        processors = processors_common + [
            structlog.dev.ConsoleRenderer(colors=True, repr_native_str=False)
        ]
    else:  # console (default)
        processors = processors_common + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    
    # Structlog konfigurieren
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(_config._log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _configure_stdlib_logging() -> None:
    """Fallback-Konfiguration für Standard-Logging"""
    global _config
    
    handlers = []
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    handlers.append(console_handler)
    
    # Optional: File Handler
    if _config._log_filename:
        log_path = Path(_config._log_filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=_config._log_filename,
            maxBytes=_config._max_size,
            backupCount=_config._backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=_config._log_level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )


class CompatibilityLogger:
    """
    Compatibility wrapper für Standard-Logger mit Structlog-ähnlicher API
    """
    def __init__(self, logger_name: str):
        self._logger = logging.getLogger(logger_name)
        self._bound_context = {}
    
    def _format_kwargs(self, msg: str, **kwargs) -> str:
        """Formatiert kwargs für Standard-Logger"""
        all_context = {**self._bound_context, **kwargs}
        if all_context:
            # Filter out reserved logging attributes
            reserved = {'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                       'filename', 'module', 'lineno', 'funcName', 'created', 'msecs', 
                       'relativeCreated', 'thread', 'threadName', 'processName', 'process'}
            safe_kwargs = {k: v for k, v in all_context.items() if k not in reserved}
            if safe_kwargs:
                context = " ".join(f"{k}={v}" for k, v in safe_kwargs.items())
                return f"{msg} [{context}]"
        return msg
    
    def bind(self, **kwargs):
        """Bindet Kontext an Logger (Compatibility-Methode)"""
        new_logger = CompatibilityLogger(self._logger.name)
        new_logger._bound_context = {**self._bound_context, **kwargs}
        return new_logger
    
    def info(self, msg: str, **kwargs):
        self._logger.info(self._format_kwargs(msg, **kwargs))
    
    def debug(self, msg: str, **kwargs):
        self._logger.debug(self._format_kwargs(msg, **kwargs))
    
    def warning(self, msg: str, **kwargs):
        self._logger.warning(self._format_kwargs(msg, **kwargs))
    
    def error(self, msg: str, **kwargs):
        self._logger.error(self._format_kwargs(msg, **kwargs))
    
    def exception(self, msg: str, **kwargs):
        self._logger.exception(self._format_kwargs(msg, **kwargs))
    
    def critical(self, msg: str, **kwargs):
        self._logger.critical(self._format_kwargs(msg, **kwargs))


def get_logger(name: str):
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
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return CompatibilityLogger(name)


def get_bound_logger(name: str, **context):
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
    logger = get_logger(name)
    return logger.bind(**context)


def get_logging_config() -> Dict[str, Any]:
    """
    Gibt die aktuelle Logging-Konfiguration zurück.
    
    Returns:
        Dictionary mit Konfigurationsinformationen
    """
    global _config
    return {
        "structlog_available": STRUCTLOG_AVAILABLE,
        "configured": _config.is_configured,
        "log_level": logging.getLevelName(_config._log_level),
        "log_format": _config._log_format,
        "log_filename": _config._log_filename,
        "max_size_mb": _config._max_size // (1024 * 1024),
        "backup_count": _config._backup_count
    }


def reconfigure_logging(**kwargs) -> None:
    """
    Rekonfiguriert das Logging-System zur Laufzeit.
    
    Args:
        **kwargs: Neue Konfigurationsparameter
            - log_level: Neues Log-Level
            - log_format: Neues Format
            - log_filename: Neue Log-Datei
    """
    global _config
    
    # Update configuration
    if 'log_level' in kwargs:
        level_name = kwargs['log_level'].upper()
        _config._log_level = getattr(logging, level_name, logging.INFO)
    
    if 'log_format' in kwargs:
        _config._log_format = kwargs['log_format'].lower()
    
    if 'log_filename' in kwargs:
        _config._log_filename = kwargs['log_filename']
    
    # Force reconfiguration
    _config._configured = False
    configure_logging()


# Automatische Konfiguration beim Import
if not _config.is_configured:
    configure_logging()


def get_logging_config() -> Dict[str, Any]:
    """
    Gibt die aktuelle Logging-Konfiguration zurück.
    
    Returns:
        Dictionary mit aktueller Logging-Konfiguration
    """
    return {
        'level': logging.getLevelName(_config._log_level),
        'format': _config._log_format,
        'filename': _config._log_filename,
        'max_size_mb': _config._max_size,
        'backup_count': _config._backup_count,
        'structlog_available': STRUCTLOG_AVAILABLE,
        'configured': _config._configured
    }


def set_log_level(level: str) -> None:
    """
    Setzt das Logging-Level zur Laufzeit.
    
    Args:
        level: Logging-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level_name = level.upper()
    if hasattr(logging, level_name):
        _config._log_level = getattr(logging, level_name)
        # Reconfigure logging
        _config._configured = False
        configure_logging()
    else:
        raise ValueError(f"Invalid log level: {level}")


def set_log_format(format_type: str) -> None:
    """
    Setzt das Logging-Format zur Laufzeit.
    
    Args:
        format_type: Format-Typ ("json", "console", "dev")
    """
    valid_formats = ['json', 'console', 'dev']
    if format_type.lower() in valid_formats:
        _config._log_format = format_type.lower()
        # Reconfigure logging
        _config._configured = False
        configure_logging()
    else:
        raise ValueError(f"Invalid log format: {format_type}. Valid options: {valid_formats}")


def get_logger_names() -> List[str]:
    """
    Gibt eine Liste aller konfigurierten Logger-Namen zurück.
    
    Returns:
        Liste der Logger-Namen
    """
    return list(logging.Logger.manager.loggerDict.keys())


def reset_logging_config() -> None:
    """
    Setzt die Logging-Konfiguration zurück und lädt sie neu.
    """
    _config._configured = False
    configure_logging()
