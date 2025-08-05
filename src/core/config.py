#!/usr/bin/env python3
"""
🔧 BU-PROCESSOR CONFIGURATION - PYDANTIC REFACTORED
==================================================
Moderne Konfiguration mit Pydantic BaseSettings für automatisches
Laden von Umgebungsvariablen, Validierung und Environment-Management.

KEY IMPROVEMENTS:
- Pydantic BaseSettings: Automatisches Environment Loading
- Pfad-Validierung: Überprüfung ob Verzeichnisse existieren
- Token/Key-Validierung: Nicht-leere Secrets
- Environment-spezifische Konfiguration
- Type Safety: Vollständige Typisierung
- Sensible Defaults: Entwickler-freundliche Standardwerte
"""

import os
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any, Union
from enum import Enum

from pydantic import BaseSettings, Field, validator, root_validator
from pydantic.types import PositiveInt, PositiveFloat, DirectoryPath

# ============================================================================
# ENUMS FÜR TYPE SAFETY
# ============================================================================

class Environment(str, Enum):
    """Verfügbare Umgebungen"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Verfügbare Log-Level"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class PDFExtractionMethod(str, Enum):
    """Verfügbare PDF-Extraktionsmethoden"""
    PYMUPDF = "pymupdf"
    PYPDF2 = "pypdf2"
    PDFPLUMBER = "pdfplumber"

# ============================================================================
# BASE CONFIGURATION CLASSES
# ============================================================================

class MLModelConfig(BaseSettings):
    """ML-Model spezifische Konfiguration"""
    
    # Model Paths
    model_path: str = Field(
        default="bert-base-german-cased",
        description="Pfad zum trainierten ML-Modell"
    )
    sentence_transformer_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        description="Sentence Transformer Model Name"
    )
    model_name: str = Field(
        default="bu-classifier-v1",
        description="Model Name für Logging"
    )
    
    # Model Parameters
    max_sequence_length: PositiveInt = Field(
        default=512,
        description="Maximale Token-Länge für das Modell"
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Schwellwert für sichere Klassifikation"
    )
    use_gpu: bool = Field(
        default=True,
        description="GPU-Nutzung aktivieren falls verfügbar"
    )
    
    @validator('model_path')
    def validate_model_path(cls, v):
        """Überprüfe ob Model-Pfad existiert oder erstellt werden kann"""
        import structlog
        logger = structlog.get_logger("config.validator")
        
        path = Path(v)
        if not path.exists():
            # Erstelle Verzeichnis falls nicht vorhanden
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Model-Pfad erstellt", path=str(path.parent))
        return v
    
    @validator('sentence_transformer_model')
    def validate_sentence_transformer_model(cls, v):
        """Validiere Sentence Transformer Model Name"""
        import structlog
        logger = structlog.get_logger("config.validator")
        
        # Gültige Modell-Präfixe und bekannte Modelle
        valid_prefixes = [
            'sentence-transformers/',
            'paraphrase-',
            'all-',
            'multi-qa-',
            'distilbert-',
            'bert-',
            'roberta-'
        ]
        
        known_models = [
            'paraphrase-multilingual-MiniLM-L12-v2',
            'paraphrase-multilingual-mpnet-base-v2',
            'paraphrase-MiniLM-L6-v2',
            'all-MiniLM-L6-v2',
            'all-mpnet-base-v2'
        ]
        
        # Prüfe ob Modell bekannt ist oder gültiges Präfix hat
        if v not in known_models and not any(v.startswith(prefix) for prefix in valid_prefixes):
            logger.warning(
                "Unbekanntes Sentence Transformer Modell",
                model=v,
                known_models=known_models[:3],  # Zeige nur erste 3
                suggestion="Verwende bekannte Modelle oder prüfe Verfügbarkeit"
            )
            # Warnung aber kein Fehler - Modell könnte trotzdem funktionieren
        
        # Prüfe auf verdächtige/ungültige Zeichen
        if not v.replace('-', '').replace('_', '').replace('/', '').replace('.', '').isalnum():
            raise ValueError(f"Sentence Transformer Model enthält ungültige Zeichen: {v}")
        
        # Mindestlänge prüfen
        if len(v) < 3:
            raise ValueError("Sentence Transformer Model Name zu kurz")
        
        return v

class PDFProcessingConfig(BaseSettings):
    """PDF-Verarbeitung spezifische Konfiguration"""
    
    # Extraction Settings
    extraction_method: PDFExtractionMethod = Field(
        default=PDFExtractionMethod.PYMUPDF,
        description="Bevorzugte PDF-Extraktionsmethode"
    )
    max_pdf_size_mb: PositiveInt = Field(
        default=50,
        description="Maximale PDF-Dateigröße in MB"
    )
    max_pdf_pages: PositiveInt = Field(
        default=100,
        description="Maximale Anzahl Seiten pro PDF"
    )
    
    # Text Processing
    text_cleanup: bool = Field(
        default=True,
        description="PDF-Text Bereinigung aktivieren"
    )
    min_extracted_text_length: PositiveInt = Field(
        default=10,
        description="Mindest-Text-Länge für gültige Extraktion"
    )
    normalize_whitespace: bool = Field(
        default=True,
        description="Leerzeichen und Zeilenumbrüche normalisieren"
    )
    auto_detect_language: bool = Field(
        default=False,
        description="Automatische Sprach-Erkennung"
    )
    
    # Batch Processing
    max_batch_pdf_count: PositiveInt = Field(
        default=20,
        description="Maximale Anzahl PDFs gleichzeitig"
    )
    fallback_chain: bool = Field(
        default=True,
        description="Text-Extraktion Fallback-Kette aktivieren"
    )
    extract_metadata: bool = Field(
        default=True,
        description="PDF-Metadaten extrahieren und speichern"
    )
    
    # Cache Settings
    cache_dir: str = Field(
        default="cache/pdf_extractions",
        description="Cache-Verzeichnis für PDF-Extraktion"
    )
    enable_cache: bool = Field(
        default=True,
        description="PDF-Cache aktivieren"
    )
    
    # File Types
    supported_extensions: List[str] = Field(
        default=[".pdf"],
        description="Unterstützte Dateierweiterungen"
    )
    
    @validator('cache_dir')
    def validate_cache_dir(cls, v):
        """Erstelle Cache-Verzeichnis falls nicht vorhanden"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

class APIConfig(BaseSettings):
    """API und Webservice Konfiguration"""
    
    # Server Settings
    host: str = Field(
        default="0.0.0.0",
        description="API Host"
    )
    port: PositiveInt = Field(
        default=8000,
        description="API Port"
    )
    
    # Monitoring
    prometheus_metrics_port: PositiveInt = Field(
        default=9100,
        description="Prometheus Metrics Port"
    )
    
    # Security (aus Umgebungsvariablen)
    api_key: Optional[str] = Field(
        default=None,
        description="API Key für geschützte Endpoints"
    )
    secret_key: Optional[str] = Field(
        default=None,
        description="Secret Key für JWT/Sessions"
    )
    
    @validator('api_key')
    def validate_api_key(cls, v, values):
        """Validiere API Key falls gesetzt oder in Production required"""
        # Prüfe ob wir in Production sind (falls Environment bereits gesetzt)
        environment = values.get('environment', os.getenv('BU_PROCESSOR_ENVIRONMENT', 'development'))
        is_production = environment == 'production' or environment == Environment.PRODUCTION
        
        # In Production ist API Key obligatorisch
        if is_production and (not v or v.strip() == ''):
            raise ValueError("API Key ist in Production-Umgebung obligatorisch")
        
        # Falls API Key gesetzt ist, validiere Format und Länge
        if v:
            v = v.strip()
            if len(v) < 8:
                raise ValueError("API Key muss mindestens 8 Zeichen lang sein")
            
            # Prüfe auf verdächtige Zeichen (sollte alphanumerisch + einige Sonderzeichen sein)
            import re
            if not re.match(r'^[A-Za-z0-9_\-\.\+\=]+$', v):
                raise ValueError("API Key enthält ungültige Zeichen")
            
            # Warne vor offensichtlich unsicheren Keys
            weak_keys = ['12345678', 'password', 'apikey123', 'test1234', 'development']
            if v.lower() in weak_keys:
                import structlog
                logger = structlog.get_logger("config.validator")
                logger.warning("Schwacher API Key erkannt", key_start=v[:3])
        
        return v

class VectorDatabaseConfig(BaseSettings):
    """Vector Database (Pinecone) Konfiguration"""
    
    # Pinecone Settings
    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="Pinecone API Key"
    )
    pinecone_environment: str = Field(
        default="us-west1-gcp-free",
        description="Pinecone Environment"
    )
    pinecone_index_name: str = Field(
        default="bu-processor",
        description="Pinecone Index Name"
    )
    
    # Embedding Settings
    embedding_dimension: PositiveInt = Field(
        default=384,
        description="Embedding Dimension"
    )
    embedding_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        description="Embedding Model Name"
    )
    
    # Processing Settings
    max_batch_size: PositiveInt = Field(
        default=100,
        description="Maximale Batch-Größe für Uploads"
    )
    enable_vector_db: bool = Field(
        default=False,
        description="Vector Database Integration aktivieren"
    )
    
    @validator('pinecone_api_key')
    def validate_pinecone_key(cls, v):
        """Validiere Pinecone API Key falls gesetzt"""
        if v and (len(v) < 10 or not v.startswith(('pc-', 'sk-'))):
            raise ValueError("Ungültiger Pinecone API Key Format")
        return v

class DeduplicationConfig(BaseSettings):
    """Deduplication und SimHash Konfiguration"""
    
    # SimHash Settings
    simhash_bit_size: PositiveInt = Field(
        default=64,
        description="Bit-Größe für SimHash-Algorithmus"
    )
    simhash_ngram_size: PositiveInt = Field(
        default=3,
        description="N-Gram-Größe für semantische Analyse"
    )
    
    # Similarity Thresholds
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Semantische Ähnlichkeitsschwelle für Duplikate"
    )
    hamming_threshold: PositiveInt = Field(
        default=8,
        description="Maximale Hamming-Distanz für Duplikate"
    )
    
    # Content-Type Weights
    content_type_weights: Dict[str, float] = Field(
        default={
            'legal_text': 1.5,
            'technical_spec': 1.3,
            'narrative': 1.0,
            'table': 0.8,
            'list': 0.7,
            'mixed': 1.0
        },
        description="Gewichtung verschiedener Content-Typen"
    )
    
    # Representative Selection Weights
    selection_weights: Dict[str, float] = Field(
        default={
            'text_length_factor': 0.001,
            'heading_level_factor': 10.0,
            'token_count_factor': 0.1
        },
        description="Gewichtung für Representative-Auswahl"
    )
    
    # Performance Settings
    enable_caching: bool = Field(
        default=True,
        description="SimHash-Caching aktivieren"
    )
    cache_size_limit: PositiveInt = Field(
        default=10000,
        description="Maximale Anzahl Cache-Einträge"
    )
    
    # Feature Extraction Settings
    enable_semantic_features: bool = Field(
        default=True,
        description="Erweiterte semantische Feature-Extraktion"
    )
    min_important_word_length: PositiveInt = Field(
        default=6,
        description="Mindestlänge für wichtige Wörter"
    )
    min_token_length: PositiveInt = Field(
        default=3,
        description="Mindestlänge für Token-Verarbeitung"
    )
    
    # Enable/Disable Flag
    enable_semantic_deduplication: bool = Field(
        default=True,
        description="Semantische Deduplication aktivieren"
    )

class SemanticConfig(BaseSettings):
    """Semantische Clustering Konfiguration"""
    
    # Model Settings
    models: Dict[str, str] = Field(
        default={
            'multilingual': 'paraphrase-multilingual-MiniLM-L12-v2',
            'german_legal': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            'fast_embedding': 'paraphrase-MiniLM-L6-v2'
        },
        description="Verfügbare Embedding-Modelle"
    )
    
    # Batch Processing Settings
    batch_processing: Dict[str, int] = Field(
        default={
            'default_batch_size': 32,
            'large_batch_size': 64,
            'small_batch_size': 16,
            'max_chunks_for_batching': 100
        },
        description="Batch-Verarbeitungsparameter"
    )
    
    # Clustering Strategies per Content Type
    clustering: Dict[str, Dict[str, Any]] = Field(
        default={
            'legal_text': {
                'algorithm': 'DBSCAN',
                'eps': 0.3,
                'min_samples': 2,
                'metric': 'cosine'
            },
            'table_heavy': {
                'algorithm': 'AgglomerativeClustering',
                'n_clusters': None,
                'distance_threshold': 0.7,
                'linkage': 'average'
            },
            'technical': {
                'algorithm': 'DBSCAN',
                'eps': 0.4,
                'min_samples': 3,
                'metric': 'cosine'
            },
            'narrative': {
                'algorithm': 'AgglomerativeClustering',
                'n_clusters': None,
                'distance_threshold': 0.6,
                'linkage': 'ward'
            },
            'mixed': {
                'algorithm': 'DBSCAN',
                'eps': 0.45,
                'min_samples': 2,
                'metric': 'cosine'
            }
        },
        description="Clustering-Strategien für verschiedene Content-Typen"
    )
    
    # Caching Settings
    caching: Dict[str, Any] = Field(
        default={
            'max_cache_size': 1000,
            'cache_ttl_seconds': 3600,
            'enable_persistent_cache': False
        },
        description="Cache-Konfiguration für Embeddings"
    )
    
    # Similarity Settings
    similarity: Dict[str, Any] = Field(
        default={
            'min_similarity_threshold': 0.3,
            'top_similar_chunks': 3
        },
        description="Ähnlichkeits-Schwellwerte und Parameter"
    )
    
    # Enable/Disable Flag
    enable_semantic_clustering: bool = Field(
        default=True,
        description="Semantisches Clustering aktivieren"
    )

class OpenAIConfig(BaseSettings):
    """OpenAI API Konfiguration"""
    
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API Key"
    )
    openai_model: str = Field(
        default="gpt-3.5-turbo",
        description="OpenAI Model für Chatbot"
    )
    max_tokens: PositiveInt = Field(
        default=1000,
        description="Maximale Tokens pro Request"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature für Text-Generation"
    )
    enable_chatbot: bool = Field(
        default=False,
        description="Chatbot Integration aktivieren"
    )
    
    @validator('openai_api_key')
    def validate_openai_key(cls, v):
        """Validiere OpenAI API Key falls gesetzt"""
        if v and not v.startswith('sk-'):
            raise ValueError("OpenAI API Key muss mit 'sk-' beginnen")
        return v

# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

class BUProcessorConfig(BaseSettings):
    """Hauptkonfiguration für BU-Processor mit Environment-Management"""
    
    # Environment Settings
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Aktuelles Environment"
    )
    debug: bool = Field(
        default=True,
        description="Debug-Modus aktivieren"
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Log-Level"
    )
    
    # Sub-Configurations
    ml_model: MLModelConfig = Field(default_factory=MLModelConfig)
    pdf_processing: PDFProcessingConfig = Field(default_factory=PDFProcessingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    vector_db: VectorDatabaseConfig = Field(default_factory=VectorDatabaseConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    
    # Application Metadata
    app_name: str = Field(
        default="BU-Processor",
        description="Anwendungsname"
    )
    version: str = Field(
        default="3.0.0",
        description="Anwendungsversion"
    )
    
    @root_validator
    def configure_environment_settings(cls, values):
        """Konfiguriere Settings basierend auf Environment"""
        env = values.get('environment', Environment.DEVELOPMENT)
        
        if env == Environment.PRODUCTION:
            # Production: Sicherheits- und Performance-optimiert
            values['debug'] = False
            values['log_level'] = LogLevel.WARNING
            
            # Reduzierte Limits für Stabilität
            if 'pdf_processing' in values:
                pdf_config = values['pdf_processing']
                pdf_config.max_pdf_size_mb = min(pdf_config.max_pdf_size_mb, 25)
                pdf_config.max_batch_pdf_count = min(pdf_config.max_batch_pdf_count, 10)
                pdf_config.enable_cache = False  # Kein Cache in Production
            
            # Zusätzliche Production-Validierung
            api_config = values.get('api')
            if api_config and not api_config.secret_key:
                import structlog
                logger = structlog.get_logger("config.validator")
                logger.warning("Production ohne SECRET_KEY - bitte setzen für sichere Sessions")
        
        elif env == Environment.STAGING:
            # Staging: Production-nah aber mit mehr Logging
            values['debug'] = False
            values['log_level'] = LogLevel.INFO
            
            if 'pdf_processing' in values:
                pdf_config = values['pdf_processing']
                pdf_config.max_pdf_size_mb = min(pdf_config.max_pdf_size_mb, 50)
        
        elif env == Environment.DEVELOPMENT:
            # Development: Maximale Flexibilität und Debugging
            values['debug'] = True
            values['log_level'] = LogLevel.DEBUG
            
            if 'pdf_processing' in values:
                pdf_config = values['pdf_processing']
                pdf_config.max_pdf_size_mb = 100  # Größere Limits für Tests
                pdf_config.max_batch_pdf_count = 50
        
        return values
    
    @validator('environment', pre=True)
    def parse_environment(cls, v):
        """Parse Environment aus String oder ENV-Variable"""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    def get_cache_dir(self) -> Path:
        """Gibt vollständigen Cache-Pfad zurück"""
        return Path(self.pdf_processing.cache_dir).resolve()
    
    def get_model_dir(self) -> Path:
        """Gibt vollständigen Model-Pfad zurück"""
        return Path(self.ml_model.model_path).parent.resolve()
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Prüft ob ein Feature aktiviert ist"""
        feature_map = {
            "vector_db": self.vector_db.enable_vector_db,
            "chatbot": self.openai.enable_chatbot,
            "cache": self.pdf_processing.enable_cache,
            "gpu": self.ml_model.use_gpu,
            "metadata_extraction": self.pdf_processing.extract_metadata,
            "semantic_clustering": self.semantic.enable_semantic_clustering,
            "semantic_deduplication": self.deduplication.enable_semantic_deduplication
        }
        return feature_map.get(feature, False)
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Gibt Environment-Informationen zurück"""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "log_level": self.log_level.value,
            "version": self.version,
            "features_enabled": {
                "vector_db": self.is_feature_enabled("vector_db"),
                "chatbot": self.is_feature_enabled("chatbot"),
                "cache": self.is_feature_enabled("cache"),
                "gpu": self.is_feature_enabled("gpu"),
                "semantic_clustering": self.is_feature_enabled("semantic_clustering"),
                "semantic_deduplication": self.is_feature_enabled("semantic_deduplication")
            }
        }
    
    class Config:
        # Environment-Variable Prefixes
        env_prefix = "BU_PROCESSOR_"
        env_nested_delimiter = "__"
        
        # Environment file loading
        env_file = ".env"
        env_file_encoding = "utf-8"
        
        # Case sensitivity
        case_sensitive = False
        
        # Allow extra fields for future extensions
        extra = "ignore"

# ============================================================================
# CONFIGURATION FACTORY
# ============================================================================

def create_config(
    environment: Optional[str] = None,
    config_file: Optional[str] = None
) -> BUProcessorConfig:
    """Factory für Konfiguration mit optionalem Environment Override"""
    
    # Setup logger
    import structlog
    logger = structlog.get_logger("config.factory")
    
    # Environment aus Parameter oder ENV-Variable
    if environment:
        os.environ["BU_PROCESSOR_ENVIRONMENT"] = environment
    
    # Config file Override
    env_file = config_file or ".env"
    
    try:
        config = BUProcessorConfig(_env_file=env_file)
        
        logger.info("Konfiguration erfolgreich geladen",
                   environment=config.environment.value,
                   log_level=config.log_level.value,
                   vector_db_enabled=config.is_feature_enabled('vector_db'),
                   chatbot_enabled=config.is_feature_enabled('chatbot'),
                   semantic_enabled=config.is_feature_enabled('semantic_clustering'),
                   deduplication_enabled=config.is_feature_enabled('semantic_deduplication'))
        
        return config
    
    except Exception as e:
        logger.error("Konfigurationsfehler", error=str(e))
        logger.info("Fallback zu Development-Konfiguration")
        
        # Fallback zu minimaler Development-Konfiguration
        return BUProcessorConfig(environment=Environment.DEVELOPMENT)

def validate_config(config: BUProcessorConfig) -> List[str]:
    """Validiert Konfiguration und gibt Liste von Problemen zurück"""
    
    issues = []
    
    # Pfad-Validierung
    cache_dir = config.get_cache_dir()
    if not cache_dir.exists():
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cache-Verzeichnis kann nicht erstellt werden: {e}")
    
    model_dir = config.get_model_dir()
    if not model_dir.exists():
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Model-Verzeichnis kann nicht erstellt werden: {e}")
    
    # API Key Validierung
    if config.is_feature_enabled("vector_db") and not config.vector_db.pinecone_api_key:
        issues.append("Vector DB aktiviert aber Pinecone API Key fehlt")
    
    if config.is_feature_enabled("chatbot") and not config.openai.openai_api_key:
        issues.append("Chatbot aktiviert aber OpenAI API Key fehlt")
    
    if config.is_feature_enabled("semantic_deduplication") and config.deduplication.similarity_threshold <= 0:
        issues.append("Semantic Deduplication aktiviert aber ungültiger Similarity Threshold")
    
    # Production-spezifische Validierung
    if config.environment == Environment.PRODUCTION:
        if not config.api.secret_key:
            issues.append("Production Environment benötigt SECRET_KEY")
        
        if config.debug:
            issues.append("Debug-Modus sollte in Production deaktiviert sein")
    
    return issues

# ============================================================================
# GLOBAL CONFIGURATION INSTANCE
# ============================================================================

# Automatische Konfiguration beim Import
try:
    settings = create_config()
    
    # Setup logger für Validierung
    import structlog
    validation_logger = structlog.get_logger("config.validation")
    
    # Validierung
    config_issues = validate_config(settings)
    if config_issues:
        validation_logger.warning("Konfigurationswarnungen gefunden", issues=config_issues)
    else:
        validation_logger.info("Konfiguration erfolgreich validiert")

except Exception as e:
    # Fallback logging ohne strukturlog
    import logging
    fallback_logger = logging.getLogger("config")
    fallback_logger.error(f"Kritischer Konfigurationsfehler: {e}")
    
    # Fallback zu minimaler Konfiguration
    settings = BUProcessorConfig(environment=Environment.DEVELOPMENT)

# ============================================================================
# CONVENIENCE EXPORTS
# ============================================================================

# Backward compatibility exports
ML_MODEL_PATH = settings.ml_model.model_path
MAX_SEQUENCE_LENGTH = settings.ml_model.max_sequence_length
CONFIDENCE_THRESHOLD = settings.ml_model.confidence_threshold
USE_GPU = settings.ml_model.use_gpu
SENTENCE_TRANSFORMER_MODEL = settings.ml_model.sentence_transformer_model

API_HOST = settings.api.host
API_PORT = settings.api.port

PDF_EXTRACTION_METHOD = settings.pdf_processing.extraction_method.value
MAX_PDF_SIZE_MB = settings.pdf_processing.max_pdf_size_mb
MAX_PDF_PAGES = settings.pdf_processing.max_pdf_pages
PDF_TEXT_CLEANUP = settings.pdf_processing.text_cleanup

# Semantic Configuration exports
SEMANTIC_CONFIG = {
    'models': settings.semantic.models,
    'batch_processing': settings.semantic.batch_processing,
    'clustering': settings.semantic.clustering,
    'caching': settings.semantic.caching,
    'similarity': settings.semantic.similarity
}

# Deduplication Configuration exports
DEDUPLICATION_CONFIG = {
    'simhash_bit_size': settings.deduplication.simhash_bit_size,
    'simhash_ngram_size': settings.deduplication.simhash_ngram_size,
    'similarity_threshold': settings.deduplication.similarity_threshold,
    'hamming_threshold': settings.deduplication.hamming_threshold,
    'content_type_weights': settings.deduplication.content_type_weights,
    'selection_weights': settings.deduplication.selection_weights,
    'enable_caching': settings.deduplication.enable_caching,
    'cache_size_limit': settings.deduplication.cache_size_limit,
    'enable_semantic_features': settings.deduplication.enable_semantic_features,
    'min_important_word_length': settings.deduplication.min_important_word_length,
    'min_token_length': settings.deduplication.min_token_length
}
SIMILARITY_THRESHOLD = settings.deduplication.similarity_threshold

LOG_LEVEL = settings.log_level.value
ENV = settings.environment.value

# Modern exports
config = settings

def get_config() -> BUProcessorConfig:
    """Gibt aktuelle Konfiguration zurück"""
    return settings

def reload_config(environment: Optional[str] = None) -> BUProcessorConfig:
    """Lädt Konfiguration neu"""
    global settings
    settings = create_config(environment)
    return settings

# ============================================================================
# DEMO FUNCTION
# ============================================================================

def demo_config():
    """Demo der modernen Konfiguration - sicher ohne eval()"""
    
    # Setup demo logger
    import structlog
    demo_logger = structlog.get_logger("config.demo")
    
    demo_logger.info("BU-Processor Configuration Demo gestartet")
    
    # Zeige aktuelle Konfiguration
    env_info = settings.get_environment_info()
    demo_logger.info("Environment Info", **env_info)
    
    # Teste verschiedene Environments - sicher ohne eval()
    demo_logger.info("Testing Different Environments")
    
    environments = {
        "development": Environment.DEVELOPMENT,
        "staging": Environment.STAGING, 
        "production": Environment.PRODUCTION
    }
    
    for env_name, env_enum in environments.items():
        demo_logger.info(f"Testing {env_name.upper()} environment")
        try:
            test_config = create_config(environment=env_name)
            
            demo_logger.info(f"{env_name.upper()} config",
                            debug=test_config.debug,
                            log_level=test_config.log_level.value,
                            max_pdf_size=f"{test_config.pdf_processing.max_pdf_size_mb}MB",
                            batch_count=test_config.pdf_processing.max_batch_pdf_count,
                            cache_enabled=test_config.pdf_processing.enable_cache)
        except Exception as e:
            demo_logger.error(f"Fehler beim Testen von {env_name}", error=str(e))
    
    # Zeige Pfade
    try:
        demo_logger.info("Wichtige Pfade",
                        cache_dir=str(settings.get_cache_dir()),
                        model_dir=str(settings.get_model_dir()))
    except Exception as e:
        demo_logger.error("Fehler beim Laden der Pfade", error=str(e))
    
    # Zeige Features - sichere Iteration
    feature_list = ["vector_db", "chatbot", "cache", "gpu", "metadata_extraction", "semantic_clustering", "semantic_deduplication"]
    feature_status = {}
    
    for feature in feature_list:
        try:
            feature_status[feature] = settings.is_feature_enabled(feature)
        except Exception as e:
            demo_logger.warning(f"Fehler beim Prüfen von Feature {feature}", error=str(e))
            feature_status[feature] = False
    
    demo_logger.info("Feature Status", **feature_status)
    
    # Validierungscheck
    try:
        issues = validate_config(settings)
        if issues:
            demo_logger.warning("Konfigurationsprobleme gefunden", issues=issues)
        else:
            demo_logger.info("Keine Konfigurationsprobleme gefunden")
    except Exception as e:
        demo_logger.error("Fehler bei Konfigurationsvalidierung", error=str(e))
    
    demo_logger.info("Configuration Demo completed!")

if __name__ == "__main__":
    demo_config()
