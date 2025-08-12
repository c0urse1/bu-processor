#!/usr/bin/env python3
"""
BU-PROCESSOR REST API - ENTERPRISE-READY FASTAPI SERVER
======================================================

REST-API Server für die BU-Processor Pipeline mit FastAPI.
Bietet HTTP-Endpunkte für PDF-Klassifikation, Text-Analyse und Batch-Verarbeitung.

Features:
- PDF Upload & Klassifikation
- Text-basierte Klassifikation  
- Batch-Verarbeitung
- Health Checks & Monitoring
- Swagger/OpenAPI Dokumentation
- Async Processing
- Error Handling & Validation
- API Key Authentication
"""

import asyncio
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union
import os
import json

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# BU-Processor Imports
try:
    from ..core.config import get_config
    from ..pipeline.classifier import RealMLClassifier, extract_text_from_pdf
    from ..pipeline.pdf_extractor import ChunkingStrategy, ContentType
    import structlog
    logger = structlog.get_logger("api.main")
except ImportError as e:
    # Fallback für direkten Import
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    try:
        from core.config import get_config
        from pipeline.classifier import RealMLClassifier, extract_text_from_pdf
        from pipeline.pdf_extractor import ChunkingStrategy, ContentType
        import structlog
        logger = structlog.get_logger("api.main")
    except ImportError:
        # Final fallback
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("api.main")
        
        # Mock config für Fallback
        class MockConfig:
            api = type('obj', (object,), {
                'host': '0.0.0.0', 
                'port': 8000, 
                'api_key': None,
                'secret_key': None
            })()
            app_name = "BU-Processor"
            version = "1.0.0"
            environment = type('obj', (object,), {'value': 'development'})()
            debug = True
            
        def get_config():
            return MockConfig()

# ============================================================================
# PYDANTIC MODELS FOR API REQUESTS/RESPONSES
# ============================================================================

class TextClassificationRequest(BaseModel):
    """Request model für Text-Klassifikation"""
    text: str = Field(..., min_length=1, max_length=50000, description="Text to classify")
    include_confidence: bool = Field(default=True, description="Include confidence score")
    include_processing_time: bool = Field(default=False, description="Include processing time")

class BatchTextRequest(BaseModel):
    """Request model für Batch-Text-Klassifikation"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to classify")
    batch_id: Optional[str] = Field(default=None, description="Optional batch identifier")

class PDFClassificationRequest(BaseModel):
    """Request model für PDF-Klassifikations-Parameter"""
    chunking_strategy: str = Field(default="simple", description="Chunking strategy: simple, semantic, hybrid")
    max_chunk_size: int = Field(default=1000, ge=100, le=5000, description="Maximum chunk size")
    classify_chunks_individually: bool = Field(default=False, description="Classify each chunk individually")

class ClassificationResponse(BaseModel):
    """Response model für Klassifikationsergebnisse"""
    category: int = Field(..., description="Predicted category ID")
    category_label: Optional[str] = Field(None, description="Human-readable category label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    is_confident: bool = Field(..., description="Whether confidence meets threshold")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    input_type: str = Field(..., description="Type of input processed")

class PDFClassificationResponse(ClassificationResponse):
    """Response model für PDF-spezifische Klassifikation"""
    file_name: str = Field(..., description="Original filename")
    page_count: int = Field(..., description="Number of pages")
    text_length: int = Field(..., description="Extracted text length")
    extraction_method: str = Field(..., description="PDF extraction method used")
    chunking_enabled: bool = Field(..., description="Whether chunking was used")

class BatchClassificationResponse(BaseModel):
    """Response model für Batch-Klassifikation"""
    batch_id: str = Field(..., description="Batch identifier")
    total_processed: int = Field(..., description="Total items processed")
    successful: int = Field(..., description="Successfully processed items")
    failed: int = Field(..., description="Failed items")
    batch_time: float = Field(..., description="Total batch processing time")
    results: List[ClassificationResponse] = Field(..., description="Individual results")

class HealthResponse(BaseModel):
    """Response model für Health Check"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Current environment")
    classifier_loaded: bool = Field(..., description="Whether ML classifier is loaded")
    features_enabled: Dict[str, bool] = Field(..., description="Enabled features")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

class ErrorResponse(BaseModel):
    """Response model für Fehler"""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request identifier")

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

# Konfiguration laden
try:
    config = get_config()
    logger.info("Configuration loaded successfully", environment=config.environment.value)
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    config = MockConfig()

# FastAPI App initialisieren
app = FastAPI(
    title=f"{config.app_name} REST API",
    description="""
    🤖 **BU-Processor REST API** - Enterprise-ready ML document classification service
    
    ## Features
    - **PDF Classification**: Upload and classify PDF documents
    - **Text Classification**: Direct text analysis
    - **Batch Processing**: Process multiple texts at once
    - **Health Monitoring**: Service health and performance metrics
    - **Semantic Chunking**: Advanced text segmentation strategies
    
    ## Authentication
    API key authentication is required for production endpoints.
    
    ## Rate Limits
    - PDF uploads: Max 10MB per file
    - Text input: Max 50,000 characters
    - Batch requests: Max 100 items per batch
    """,
    version=getattr(config, 'version', '1.0.0'),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if getattr(config, 'debug', True) else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE & DEPENDENCIES
# ============================================================================

# Global state
startup_time = time.time()
classifier: Optional[RealMLClassifier] = None
security = HTTPBearer(auto_error=False)

# ============================================================================
# AUTHENTICATION
# ============================================================================

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key authentication"""
    if not hasattr(config, 'api') or not config.api.api_key:
        # Keine API-Key Konfiguration - öffentlicher Zugang
        return True
    
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="API key required. Include 'Authorization: Bearer YOUR_API_KEY' header."
        )
    
    if credentials.credentials != config.api.api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return True

# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global classifier
    
    logger.info("Starting BU-Processor API server...")
    
    try:
        # Classifier initialisieren
        logger.info("Initializing ML classifier...")
        classifier = RealMLClassifier(
            batch_size=16,
            max_retries=2,
            timeout_seconds=30.0
        )
        
        # Health check
        health_status = classifier.get_health_status()
        if health_status["status"] == "healthy":
            logger.info("ML classifier loaded successfully", 
                       device=health_status.get("device"),
                       model_info=health_status.get("model_info", {}))
        else:
            logger.error("ML classifier health check failed", status=health_status)
            
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}", exc_info=True)
        classifier = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down BU-Processor API server...")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_classifier() -> RealMLClassifier:
    """Dependency to get classifier instance"""
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="ML classifier not available. Please check service health."
        )
    return classifier

def parse_chunking_strategy(strategy_str: str) -> ChunkingStrategy:
    """Parse string to ChunkingStrategy enum"""
    strategy_map = {
        "none": ChunkingStrategy.NONE,
        "simple": ChunkingStrategy.SIMPLE,
        "semantic": ChunkingStrategy.SEMANTIC,
        "hybrid": ChunkingStrategy.HYBRID,
        "balanced": ChunkingStrategy.BALANCED
    }
    return strategy_map.get(strategy_str.lower(), ChunkingStrategy.SIMPLE)

def create_error_response(error_msg: str, error_type: str, detail: str = None) -> Dict:
    """Create standardized error response"""
    return {
        "error": error_msg,
        "error_type": error_type,
        "detail": detail,
        "request_id": str(uuid.uuid4())
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": f"🤖 {config.app_name} REST API",
        "version": getattr(config, 'version', '1.0.0'),
        "docs": "/docs",
        "health": "/health",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    uptime = time.time() - startup_time
    
    # Classifier status
    classifier_loaded = classifier is not None
    classifier_status = "unknown"
    
    if classifier_loaded:
        try:
            health_status = classifier.get_health_status()
            classifier_status = health_status.get("status", "unknown")
        except Exception as e:
            logger.warning(f"Classifier health check failed: {e}")
            classifier_status = "unhealthy"
    
    # Features status (mit Fallback)
    features_enabled = {}
    try:
        if hasattr(config, 'is_feature_enabled'):
            features_enabled = {
                "vector_db": config.is_feature_enabled("vector_db"),
                "chatbot": config.is_feature_enabled("chatbot"),
                "cache": config.is_feature_enabled("cache"),
                "gpu": config.is_feature_enabled("gpu"),
                "semantic_clustering": config.is_feature_enabled("semantic_clustering"),
                "semantic_deduplication": config.is_feature_enabled("semantic_deduplication")
            }
        else:
            features_enabled = {"basic_classification": True}
    except Exception:
        features_enabled = {"basic_classification": True}
    
    return HealthResponse(
        status="healthy" if classifier_loaded and classifier_status == "healthy" else 
               "degraded" if classifier_loaded and classifier_status == "degraded" else 
               "degraded",
        version=getattr(config, 'version', '1.0.0'),
        environment=getattr(config.environment, 'value', 'unknown'),
        classifier_loaded=classifier_loaded,
        features_enabled=features_enabled,
        uptime_seconds=uptime
    )

@app.post("/classify/text", response_model=ClassificationResponse)
async def classify_text(
    request: TextClassificationRequest,
    ml_classifier: RealMLClassifier = Depends(get_classifier),
    authenticated: bool = Depends(verify_api_key)
):
    """Classify a single text input"""
    try:
        logger.info("Text classification request", text_length=len(request.text))
        
        start_time = time.time()
        result = ml_classifier.classify_text(request.text)
        processing_time = time.time() - start_time
        
        # Konvertiere Pydantic result zu dict falls nötig
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        response_data = {
            "category": result_data["category"],
            "category_label": result_data.get("category_label"),
            "confidence": result_data["confidence"],
            "is_confident": result_data["is_confident"],
            "input_type": "text"
        }
        
        if request.include_processing_time:
            response_data["processing_time"] = processing_time
        
        logger.info("Text classification completed", 
                   category=response_data["category"],
                   confidence=response_data["confidence"])
        
        return ClassificationResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Text classification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Text classification failed",
                "ClassificationError",
                str(e)
            )
        )

@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch_texts(
    request: BatchTextRequest,
    ml_classifier: RealMLClassifier = Depends(get_classifier),
    authenticated: bool = Depends(verify_api_key)
):
    """Classify multiple texts in a batch"""
    try:
        batch_id = request.batch_id or f"batch_{int(time.time())}"
        logger.info("Batch classification request", 
                   batch_id=batch_id, text_count=len(request.texts))
        
        result = ml_classifier.classify_batch(request.texts, batch_id=batch_id)
        
        # Konvertiere result zu dict falls nötig
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        # Konvertiere einzelne Results
        processed_results = []
        for individual_result in result_data["results"]:
            if hasattr(individual_result, 'dict'):
                result_dict = individual_result.dict()
            else:
                result_dict = individual_result
            
            # Erstelle ClassificationResponse für jeden Result
            if "error" not in result_dict:
                processed_results.append(ClassificationResponse(
                    category=result_dict["category"],
                    category_label=result_dict.get("category_label"),
                    confidence=result_dict["confidence"],
                    is_confident=result_dict["is_confident"],
                    processing_time=result_dict.get("processing_time"),
                    input_type=result_dict.get("input_type", "text_batch")
                ))
            else:
                # Fehler-Fall
                processed_results.append(ClassificationResponse(
                    category=-1,
                    category_label="Error",
                    confidence=0.0,
                    is_confident=False,
                    input_type="text_batch_error"
                ))
        
        response = BatchClassificationResponse(
            batch_id=batch_id,
            total_processed=result_data["total_processed"],
            successful=result_data["successful"],
            failed=result_data["failed"],
            batch_time=result_data["batch_time"],
            results=processed_results
        )
        
        logger.info("Batch classification completed", 
                   batch_id=batch_id,
                   successful=response.successful,
                   failed=response.failed)
        
        return response
        
    except Exception as e:
        logger.error(f"Batch classification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Batch classification failed",
                "BatchClassificationError",
                str(e)
            )
        )

@app.post("/classify/pdf", response_model=PDFClassificationResponse)
async def classify_pdf(
    file: UploadFile = File(...),
    chunking_strategy: str = "simple",
    max_chunk_size: int = 1000,
    classify_chunks_individually: bool = False,
    ml_classifier: RealMLClassifier = Depends(get_classifier),
    authenticated: bool = Depends(verify_api_key)
):
    """Upload and classify a PDF document"""
    
    # Validate file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Check file size (10MB limit)
    file_size = 0
    contents = await file.read()
    file_size = len(contents)
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 10MB."
        )
    
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(contents)
        tmp_path = tmp_file.name
    
    try:
        logger.info("PDF classification request", 
                   filename=file.filename, 
                   file_size=file_size,
                   chunking_strategy=chunking_strategy)
        
        # Parse chunking strategy
        strategy = parse_chunking_strategy(chunking_strategy)
        
        # Classify PDF
        result = ml_classifier.classify_pdf(
            tmp_path,
            chunking_strategy=strategy,
            max_chunk_size=max_chunk_size,
            classify_chunks_individually=classify_chunks_individually
        )
        
        # Konvertiere result zu dict falls nötig
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        if "error" in result_data:
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    "PDF processing failed",
                    "PDFProcessingError",
                    result_data["error"]
                )
            )
        
        # Erstelle Response
        response = PDFClassificationResponse(
            category=result_data["category"],
            category_label=result_data.get("category_label"),
            confidence=result_data["confidence"],
            is_confident=result_data["is_confident"],
            processing_time=result_data.get("processing_time"),
            input_type=result_data.get("input_type", "pdf"),
            file_name=file.filename,
            page_count=result_data.get("page_count", 0),
            text_length=result_data.get("text_length", len(result_data.get("text", ""))),
            extraction_method=result_data.get("extraction_method", "unknown"),
            chunking_enabled=result_data.get("chunking_enabled", False)
        )
        
        logger.info("PDF classification completed",
                   filename=file.filename,
                   category=response.category,
                   confidence=response.confidence,
                   pages=response.page_count)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF classification failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "PDF classification failed",
                "PDFClassificationError", 
                str(e)
            )
        )
    finally:
        # Cleanup temporary file
        try:
            os.unlink(tmp_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {tmp_path}: {e}")

@app.get("/models/info")
async def get_model_info(
    ml_classifier: RealMLClassifier = Depends(get_classifier),
    authenticated: bool = Depends(verify_api_key)
):
    """Get information about the loaded ML model"""
    try:
        model_info = ml_classifier.get_model_info()
        health_status = ml_classifier.get_health_status()
        
        return {
            "model_info": model_info,
            "health": health_status,
            "available_labels": ml_classifier.get_available_labels(),
            "supported_chunking_strategies": [strategy.value for strategy in ChunkingStrategy]
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Failed to retrieve model information",
                "ModelInfoError",
                str(e)
            )
        )

# ============================================================================
# MAIN FUNCTION FOR RUNNING THE SERVER
# ============================================================================

def main():
    """Main function to run the API server"""
    host = getattr(config.api, 'host', '0.0.0.0') if hasattr(config, 'api') else '0.0.0.0'
    port = getattr(config.api, 'port', 8000) if hasattr(config, 'api') else 8000
    debug = getattr(config, 'debug', True)
    
    logger.info("Starting BU-Processor API server", host=host, port=port, debug=debug)
    
    uvicorn.run(
        "bu_processor.api.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug",
        access_log=True
    )

if __name__ == "__main__":
    main()
