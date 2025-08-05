#!/usr/bin/env python3
"""
📊 ML VS. HEURISTIC EVALUATOR - PHASE 1.4 (IMPROVED)
====================================================

Comprehensive evaluation system comparing ML classifier against legacy heuristic.
Features: Centralized configuration, robust error handling, enhanced monitoring,
production-ready deployment recommendations.

IMPROVEMENTS:
- Centralized configuration management
- Robust error handling with detailed logging
- Enhanced model loading with validation
- Improved threshold optimization
- Production-ready monitoring hooks
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Core ML libraries
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# CLI & UI
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich import print as rprint
from pydantic import BaseModel, validator, Extra

# Data handling
from datasets import Dataset as HFDataset
import structlog

# =============================================================================
# CENTRALIZED CONFIGURATION
# =============================================================================

class MLEvaluationSettings:
    """Centralized configuration for ML evaluation system"""
    
    # Model paths
    ML_MODEL_PATH: str = os.getenv("ML_MODEL_PATH", "trained_models")
    TEST_DATA_PATH: str = os.getenv("TEST_DATA_PATH", "ml_training_data/test_dataset.json")
    BACKUP_MODEL_PATH: str = os.getenv("BACKUP_MODEL_PATH", "backup_models")
    
    # Model configuration
    MAX_SEQUENCE_LENGTH: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    
    # Confidence thresholds
    ML_CONFIDENCE_THRESHOLD: float = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.7"))
    MIN_CONFIDENCE_THRESHOLD: float = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.5"))
    MAX_CONFIDENCE_THRESHOLD: float = float(os.getenv("MAX_CONFIDENCE_THRESHOLD", "0.95"))
    THRESHOLD_STEP: float = float(os.getenv("THRESHOLD_STEP", "0.05"))
    
    # Performance thresholds
    MIN_ACCURACY_THRESHOLD: float = float(os.getenv("MIN_ACCURACY_THRESHOLD", "0.8"))
    MIN_F1_THRESHOLD: float = float(os.getenv("MIN_F1_THRESHOLD", "0.75"))
    PERFORMANCE_DEGRADATION_THRESHOLD: float = float(os.getenv("PERFORMANCE_DEGRADATION_THRESHOLD", "0.05"))
    
    # Output settings
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "evaluation_results")
    ENABLE_VISUALIZATIONS: bool = os.getenv("ENABLE_VISUALIZATIONS", "true").lower() == "true"
    SAVE_DETAILED_RESULTS: bool = os.getenv("SAVE_DETAILED_RESULTS", "true").lower() == "true"
    
    # Monitoring and logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_PERFORMANCE_MONITORING: bool = os.getenv("ENABLE_PERFORMANCE_MONITORING", "false").lower() == "true"
    MONITORING_INTERVAL: int = int(os.getenv("MONITORING_INTERVAL", "100"))
    
    # GPU settings
    FORCE_CPU: bool = os.getenv("FORCE_CPU", "false").lower() == "true"
    GPU_MEMORY_FRACTION: float = float(os.getenv("GPU_MEMORY_FRACTION", "0.8"))
    
    @classmethod
    def get_confidence_thresholds(cls) -> List[float]:
        """Generate confidence thresholds based on configuration"""
        thresholds = []
        current = cls.MIN_CONFIDENCE_THRESHOLD
        while current <= cls.MAX_CONFIDENCE_THRESHOLD:
            thresholds.append(round(current, 2))
            current += cls.THRESHOLD_STEP
        return thresholds
    
    @classmethod
    def validate_paths(cls) -> Dict[str, bool]:
        """Validate all configured paths"""
        return {
            "ml_model_path": Path(cls.ML_MODEL_PATH).exists(),
            "test_data_path": Path(cls.TEST_DATA_PATH).exists(),
            "backup_model_path": Path(cls.BACKUP_MODEL_PATH).exists(),
            "output_dir": True  # Will be created if not exists
        }

# Initialize settings
settings = MLEvaluationSettings()

# Initialize components
console = Console()
logger = structlog.get_logger("ml_vs_heuristic_evaluator")

# =============================================================================
# ENHANCED ERROR HANDLING
# =============================================================================

class MLEvaluationError(Exception):
    """Base exception for ML evaluation errors"""
    pass

class ModelLoadingError(MLEvaluationError):
    """Error during model loading"""
    pass

class DataLoadingError(MLEvaluationError):
    """Error during data loading"""
    pass

class EvaluationError(MLEvaluationError):
    """Error during evaluation process"""
    pass

class ConfigurationError(MLEvaluationError):
    """Error in configuration"""
    pass

def handle_exceptions(func):
    """Decorator for comprehensive exception handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ModelLoadingError as e:
            logger.error("Model loading failed", error=str(e), function=func.__name__)
            console.print(f"❌ [bold red]Model Loading Error:[/bold red] {e}")
            raise
        except DataLoadingError as e:
            logger.error("Data loading failed", error=str(e), function=func.__name__)
            console.print(f"❌ [bold red]Data Loading Error:[/bold red] {e}")
            raise
        except EvaluationError as e:
            logger.error("Evaluation failed", error=str(e), function=func.__name__)
            console.print(f"❌ [bold red]Evaluation Error:[/bold red] {e}")
            raise
        except ConfigurationError as e:
            logger.error("Configuration error", error=str(e), function=func.__name__)
            console.print(f"❌ [bold red]Configuration Error:[/bold red] {e}")
            raise
        except Exception as e:
            logger.error("Unexpected error", error=str(e), function=func.__name__, type=type(e).__name__)
            console.print(f"❌ [bold red]Unexpected Error:[/bold red] {e}")
            raise MLEvaluationError(f"Unexpected error in {func.__name__}: {e}") from e
    return wrapper

# =============================================================================
# ENHANCED CONFIGURATION CLASSES
# =============================================================================

class EvaluationConfig(BaseModel, extra=Extra.forbid):
    """Enhanced configuration for ML vs Heuristic evaluation"""
    
    # Model paths (from centralized settings)
    ml_model_path: str = settings.ML_MODEL_PATH
    test_data_path: str = settings.TEST_DATA_PATH
    backup_model_path: str = settings.BACKUP_MODEL_PATH
    
    # Evaluation settings
    confidence_thresholds: List[float] = field(default_factory=settings.get_confidence_thresholds)
    hybrid_threshold: float = settings.ML_CONFIDENCE_THRESHOLD
    max_sequence_length: int = settings.MAX_SEQUENCE_LENGTH
    batch_size: int = settings.BATCH_SIZE
    
    # Performance thresholds
    min_accuracy_threshold: float = settings.MIN_ACCURACY_THRESHOLD
    min_f1_threshold: float = settings.MIN_F1_THRESHOLD
    performance_degradation_threshold: float = settings.PERFORMANCE_DEGRADATION_THRESHOLD
    
    # Output settings
    output_dir: str = settings.OUTPUT_DIR
    create_visualizations: bool = settings.ENABLE_VISUALIZATIONS
    save_detailed_results: bool = settings.SAVE_DETAILED_RESULTS
    
    # Monitoring settings
    enable_monitoring: bool = settings.ENABLE_PERFORMANCE_MONITORING
    monitoring_interval: int = settings.MONITORING_INTERVAL
    
    # Display settings
    show_progress: bool = True
    verbose: bool = True
    
    @validator("confidence_thresholds")
    def validate_thresholds(cls, v):
        if not v:
            raise ValueError("At least one confidence threshold must be specified")
        if any(t < 0 or t > 1 for t in v):
            raise ValueError("Confidence thresholds must be between 0 and 1")
        return sorted(v)
    
    @validator("hybrid_threshold")
    def validate_hybrid_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Hybrid threshold must be between 0 and 1")
        return v
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Comprehensive configuration validation"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "path_checks": settings.validate_paths()
        }
        
        # Check critical paths
        if not validation_results["path_checks"]["ml_model_path"]:
            validation_results["errors"].append(f"ML model path not found: {self.ml_model_path}")
            validation_results["valid"] = False
        
        if not validation_results["path_checks"]["test_data_path"]:
            validation_results["errors"].append(f"Test data path not found: {self.test_data_path}")
            validation_results["valid"] = False
        
        # Check thresholds
        if self.hybrid_threshold not in self.confidence_thresholds:
            validation_results["warnings"].append(
                f"Hybrid threshold {self.hybrid_threshold} not in confidence thresholds list"
            )
        
        # Check performance thresholds
        if self.min_accuracy_threshold > 1.0:
            validation_results["errors"].append("Minimum accuracy threshold cannot exceed 1.0")
            validation_results["valid"] = False
        
        return validation_results

# Legacy integration (simulated - enhanced with better error handling)
class LegacySemanticCategorizer:
    """Enhanced legacy heuristic categorizer with robust error handling"""
    
    def __init__(self):
        self.categories = {
            0: "medical_professionals",
            1: "technical_careers", 
            2: "legal_professionals",
            3: "manual_labor",
            4: "office_workers",
            5: "education_sector",
            6: "creative_industries",
            7: "sales_marketing",
            8: "financial_services",
            9: "hospitality_service",
            10: "transport_logistics",
            11: "public_sector"
        }
        
        # Enhanced keyword patterns with better coverage
        self.keyword_patterns = {
            0: ["arzt", "ärztin", "medizin", "chirurg", "zahnarzt", "therapeut", "krankenschwester", "pfleger", "physiotherapeut"],
            1: ["ingenieur", "software", "entwickler", "programmierer", "it", "technik", "informatik", "data", "ki", "artificial"],
            2: ["anwalt", "rechtsanwalt", "jurist", "richter", "notar", "recht", "kanzlei", "rechtsbeistand"],
            3: ["handwerker", "mechaniker", "elektriker", "bauarbeiter", "monteur", "schlosser", "tischler", "maler"],
            4: ["büro", "verwaltung", "sekretär", "buchhalter", "sachbearbeiter", "assistent", "koordinator"],
            5: ["lehrer", "professor", "pädagoge", "dozent", "erzieher", "bildung", "schule", "universität"],
            6: ["künstler", "designer", "fotograf", "musiker", "autor", "kreativ", "grafiker", "illustrator"],
            7: ["verkauf", "vertrieb", "marketing", "berater", "verkäufer", "kundenservice", "akquise"],
            8: ["bank", "versicherung", "finanz", "berater", "anlage", "kredit", "investment", "börse"],
            9: ["hotel", "restaurant", "service", "kellner", "koch", "tourismus", "gastronomie", "rezeption"],
            10: ["transport", "logistik", "fahrer", "spedition", "lager", "lieferung", "versand"],
            11: ["beamter", "verwaltung", "öffentlich", "gemeinde", "staat", "behörde", "ministerium"]
        }
        
        self.performance_stats = {
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "prediction_times": []
        }
    
    @handle_exceptions
    def categorize_chunk(self, text: str, return_confidence: bool = False) -> Union[int, Tuple[int, float]]:
        """Enhanced heuristic categorization with comprehensive error handling"""
        
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text for heuristic categorization", text_type=type(text))
            if return_confidence:
                return 4, 0.1  # Default to office_workers with low confidence
            return 4
        
        start_time = time.time()
        
        try:
            text_lower = text.lower().strip()
            
            if not text_lower:
                logger.warning("Empty text after preprocessing")
                if return_confidence:
                    return 4, 0.1
                return 4
            
            scores = {}
            
            # Calculate enhanced keyword match scores
            for category_id, keywords in self.keyword_patterns.items():
                score = 0
                matched_keywords = []
                
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 1
                        matched_keywords.append(keyword)
                        
                        # Boost score for exact word matches
                        if f" {keyword} " in f" {text_lower} ":
                            score += 0.5
                        
                        # Boost for keyword at beginning or end
                        if text_lower.startswith(keyword) or text_lower.endswith(keyword):
                            score += 0.3
                
                # Normalize by number of keywords in category with bonus for multiple matches
                normalized_score = score / len(keywords)
                if len(matched_keywords) > 1:
                    normalized_score *= 1.2  # Bonus for multiple keyword matches
                
                scores[category_id] = normalized_score
                
                if settings.ENABLE_PERFORMANCE_MONITORING and matched_keywords:
                    logger.debug("Keyword matches found", 
                               category=self.categories[category_id],
                               keywords=matched_keywords,
                               score=normalized_score)
            
            # Find best match with enhanced logic
            if not scores or max(scores.values()) == 0:
                predicted_category = 4  # office_workers as default
                confidence = 0.1
                logger.debug("No keyword matches, using default category", category=predicted_category)
            else:
                # Get top matches
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                predicted_category = sorted_scores[0][0]
                confidence = min(sorted_scores[0][1], 1.0)
                
                # Adjust confidence based on score difference to second place
                if len(sorted_scores) > 1:
                    score_diff = sorted_scores[0][1] - sorted_scores[1][1]
                    if score_diff > 0.3:
                        confidence = min(confidence * 1.2, 1.0)  # Boost confidence for clear winner
                    elif score_diff < 0.1:
                        confidence *= 0.8  # Reduce confidence for close call
            
            # Update performance statistics
            prediction_time = (time.time() - start_time) * 1000
            self.performance_stats["total_predictions"] += 1
            self.performance_stats["prediction_times"].append(prediction_time)
            
            # Keep only last 1000 prediction times for rolling average
            if len(self.performance_stats["prediction_times"]) > 1000:
                self.performance_stats["prediction_times"] = self.performance_stats["prediction_times"][-1000:]
            
            # Update rolling average confidence
            current_avg = self.performance_stats["avg_confidence"]
            total = self.performance_stats["total_predictions"]
            self.performance_stats["avg_confidence"] = (current_avg * (total - 1) + confidence) / total
            
            if return_confidence:
                return predicted_category, confidence
            return predicted_category
            
        except Exception as e:
            logger.error("Error in heuristic categorization", error=str(e), text_preview=text[:100])
            # Return safe defaults
            if return_confidence:
                return 4, 0.1
            return 4
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = self.performance_stats.copy()
        if stats["prediction_times"]:
            stats["avg_prediction_time_ms"] = np.mean(stats["prediction_times"])
            stats["median_prediction_time_ms"] = np.median(stats["prediction_times"])
            stats["p95_prediction_time_ms"] = np.percentile(stats["prediction_times"], 95)
        else:
            stats.update({
                "avg_prediction_time_ms": 0,
                "median_prediction_time_ms": 0,
                "p95_prediction_time_ms": 0
            })
        return stats

# =============================================================================
# ENHANCED ML MODEL LOADER
# =============================================================================

class MLModelLoader:
    """Enhanced ML model loader with comprehensive error handling and validation"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.backup_path = Path(settings.BACKUP_MODEL_PATH) if hasattr(settings, 'BACKUP_MODEL_PATH') else None
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.model_info = {}
        self.performance_stats = {
            "total_predictions": 0,
            "total_inference_time": 0,
            "batch_sizes": [],
            "memory_usage": []
        }
        
    def _get_device(self) -> torch.device:
        """Determine optimal device with user preferences"""
        if settings.FORCE_CPU:
            logger.info("Forcing CPU usage as per configuration")
            return torch.device('cpu')
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # Set memory fraction if configured
            if hasattr(settings, 'GPU_MEMORY_FRACTION'):
                torch.cuda.set_per_process_memory_fraction(settings.GPU_MEMORY_FRACTION)
            logger.info("Using CUDA device", device=str(device), memory_fraction=settings.GPU_MEMORY_FRACTION)
            return device
        else:
            logger.info("CUDA not available, using CPU")
            return torch.device('cpu')
    
    @handle_exceptions
    def validate_model_files(self, model_path: Path) -> Tuple[bool, List[str]]:
        """Comprehensive model file validation"""
        if not model_path.exists():
            return False, [f"Model directory does not exist: {model_path}"]
        
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer_config.json",
            "vocab.txt"  # or tokenizer.json for newer models
        ]
        
        optional_files = [
            "tokenizer.json",
            "special_tokens_map.json",
            "training_args.bin"
        ]
        
        missing_required = []
        missing_optional = []
        
        for file in required_files:
            if not (model_path / file).exists():
                # Check alternative names
                if file == "pytorch_model.bin" and (model_path / "pytorch_model.safetensors").exists():
                    continue
                if file == "vocab.txt" and (model_path / "tokenizer.json").exists():
                    continue
                missing_required.append(file)
        
        for file in optional_files:
            if not (model_path / file).exists():
                missing_optional.append(file)
        
        errors = []
        if missing_required:
            errors.append(f"Missing required files: {missing_required}")
        
        if missing_optional:
            logger.warning("Missing optional files", files=missing_optional)
        
        return len(errors) == 0, errors
    
    @handle_exceptions
    def load_model(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """Enhanced model loading with fallback and validation"""
        
        logger.info("Starting model loading process", model_path=str(self.model_path))
        
        # Primary model loading attempt
        is_valid, errors = self.validate_model_files(self.model_path)
        if not is_valid:
            error_msg = f"Model validation failed: {'; '.join(errors)}"
            
            # Try backup model if available
            if self.backup_path and self.backup_path.exists():
                logger.warning("Primary model invalid, trying backup", 
                             primary_path=str(self.model_path),
                             backup_path=str(self.backup_path))
                
                backup_valid, backup_errors = self.validate_model_files(self.backup_path)
                if backup_valid:
                    self.model_path = self.backup_path
                    logger.info("Using backup model", path=str(self.backup_path))
                else:
                    raise ModelLoadingError(f"Both primary and backup models invalid. Primary: {error_msg}. Backup: {'; '.join(backup_errors)}")
            else:
                raise ModelLoadingError(error_msg)
        
        try:
            # Load tokenizer and model with centralized configuration
            logger.info("Loading tokenizer", path=str(self.model_path))
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    settings.ML_MODEL_PATH,
                    use_fast=True,  # Use fast tokenizer when available
                    trust_remote_code=False  # Security: don't trust remote code
                )
                logger.info("Tokenizer loaded successfully", 
                           vocab_size=self.tokenizer.vocab_size,
                           model_max_length=getattr(self.tokenizer, 'model_max_length', 'unknown'))
            except Exception as e:
                logger.error("ML model loading failed", error=str(e))
                raise ModelLoadingError(f"ML model could not be loaded from path {settings.ML_MODEL_PATH}") from e
            
            # Load model with enhanced error handling
            logger.info("Loading ML model", path=str(self.model_path), device=str(self.device))
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    settings.ML_MODEL_PATH,
                    trust_remote_code=False,  # Security: don't trust remote code
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,  # Optimize for GPU
                    device_map=None  # Let us handle device placement
                )
                
                # Move to device
                self.model.to(self.device)
                self.model.eval()
                
                # Store model information
                self.model_info = {
                    "model_type": self.model.config.model_type,
                    "num_labels": self.model.config.num_labels,
                    "hidden_size": getattr(self.model.config, 'hidden_size', 'unknown'),
                    "num_parameters": sum(p.numel() for p in self.model.parameters()),
                    "device": str(self.device),
                    "dtype": str(next(self.model.parameters()).dtype)
                }
                
                logger.info("ML model loaded successfully", **self.model_info)
                
            except Exception as e:
                logger.error("ML model loading failed", error=str(e))
                raise ModelLoadingError(f"ML model could not be loaded from path {settings.ML_MODEL_PATH}") from e
            
            # Validation test
            try:
                self._validate_model_functionality()
                logger.info("Model functionality validation passed")
            except Exception as e:
                logger.error("Model functionality validation failed", error=str(e))
                raise ModelLoadingError(f"Model functionality validation failed: {e}") from e
            
            return self.model, self.tokenizer
            
        except ModelLoadingError:
            raise
        except Exception as e:
            raise ModelLoadingError(f"Unexpected error during model loading: {e}") from e
    
    def _validate_model_functionality(self):
        """Test model with dummy input to ensure functionality"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not loaded")
        
        test_text = "Test input for model validation"
        
        with torch.no_grad():
            inputs = self.tokenizer(
                test_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=settings.MAX_SEQUENCE_LENGTH
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            
            if outputs.logits.shape[1] != self.model.config.num_labels:
                raise RuntimeError(f"Model output shape mismatch. Expected {self.model.config.num_labels} labels, got {outputs.logits.shape[1]}")
    
    @handle_exceptions
    def predict_batch(self, texts: List[str], max_length: int = None) -> Tuple[List[int], List[float]]:
        """Enhanced batch prediction with monitoring and error handling"""
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not texts:
            logger.warning("Empty text batch provided")
            return [], []
        
        if max_length is None:
            max_length = settings.MAX_SEQUENCE_LENGTH
        
        start_time = time.time()
        batch_predictions = []
        batch_confidences = []
        failed_predictions = 0
        
        logger.info("Starting batch prediction", batch_size=len(texts), max_length=max_length)
        
        try:
            with torch.no_grad():
                for i, text in enumerate(texts):
                    try:
                        if not text or not isinstance(text, str):
                            logger.warning("Invalid text input", index=i, text_type=type(text))
                            batch_predictions.append(4)  # Default category
                            batch_confidences.append(0.1)
                            failed_predictions += 1
                            continue
                        
                        # Tokenize with enhanced error handling
                        try:
                            inputs = self.tokenizer(
                                text,
                                return_tensors="pt",
                                truncation=True,
                                padding=True,
                                max_length=max_length
                            )
                        except Exception as e:
                            logger.error("Tokenization failed", text_preview=text[:100], error=str(e))
                            batch_predictions.append(4)
                            batch_confidences.append(0.1)
                            failed_predictions += 1
                            continue
                        
                        # Move to device
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # Predict
                        outputs = self.model(**inputs)
                        
                        # Get probabilities
                        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        
                        # Get prediction and confidence
                        confidence, predicted_class = torch.max(probabilities, dim=-1)
                        
                        batch_predictions.append(predicted_class.item())
                        batch_confidences.append(confidence.item())
                        
                        # Periodic logging for large batches
                        if settings.ENABLE_PERFORMANCE_MONITORING and (i + 1) % settings.MONITORING_INTERVAL == 0:
                            elapsed = time.time() - start_time
                            rate = (i + 1) / elapsed
                            logger.info("Batch prediction progress", 
                                      completed=i+1, 
                                      total=len(texts),
                                      rate_per_sec=rate,
                                      failed=failed_predictions)
                        
                    except Exception as e:
                        logger.error("Individual prediction failed", index=i, error=str(e))
                        batch_predictions.append(4)
                        batch_confidences.append(0.1)
                        failed_predictions += 1
                        continue
            
            total_time = time.time() - start_time
            
            # Update performance statistics
            self.performance_stats["total_predictions"] += len(texts)
            self.performance_stats["total_inference_time"] += total_time
            self.performance_stats["batch_sizes"].append(len(texts))
            
            # Memory monitoring on GPU
            if self.device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
                self.performance_stats["memory_usage"].append(memory_used)
                
                if len(self.performance_stats["memory_usage"]) > 100:
                    self.performance_stats["memory_usage"] = self.performance_stats["memory_usage"][-100:]
            
            logger.info("Batch prediction completed", 
                       batch_size=len(texts),
                       failed_predictions=failed_predictions,
                       total_time_sec=total_time,
                       avg_time_per_sample_ms=(total_time * 1000) / len(texts) if texts else 0)
            
            if failed_predictions > 0:
                logger.warning("Some predictions failed", 
                              failed_count=failed_predictions,
                              success_rate=(len(texts) - failed_predictions) / len(texts) * 100)
            
            return batch_predictions, batch_confidences
            
        except Exception as e:
            logger.error("Batch prediction failed", error=str(e), batch_size=len(texts))
            raise EvaluationError(f"Batch prediction failed: {e}") from e
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = self.performance_stats.copy()
        
        if stats["total_predictions"] > 0 and stats["total_inference_time"] > 0:
            stats["avg_predictions_per_sec"] = stats["total_predictions"] / stats["total_inference_time"]
            stats["avg_time_per_prediction_ms"] = (stats["total_inference_time"] * 1000) / stats["total_predictions"]
        
        if stats["batch_sizes"]:
            stats["avg_batch_size"] = np.mean(stats["batch_sizes"])
            stats["max_batch_size"] = max(stats["batch_sizes"])
        
        if stats["memory_usage"]:
            stats["avg_memory_usage_mb"] = np.mean(stats["memory_usage"])
            stats["max_memory_usage_mb"] = max(stats["memory_usage"])
        
        stats["model_info"] = self.model_info
        
        return stats

# =============================================================================
# ENHANCED EVALUATION RESULT CLASSES
# =============================================================================

@dataclass
class EvaluationResult:
    """Enhanced results from model evaluation with additional metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    predictions: List[int]
    confidences: List[float]
    true_labels: List[int]
    inference_time_ms: float
    
    # Additional metrics
    failed_predictions: int = 0
    confidence_distribution: Dict[str, float] = field(default_factory=dict)
    per_class_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate additional metrics after initialization"""
        if self.confidences:
            self.confidence_distribution = {
                "mean": float(np.mean(self.confidences)),
                "std": float(np.std(self.confidences)),
                "min": float(np.min(self.confidences)),
                "max": float(np.max(self.confidences)),
                "p25": float(np.percentile(self.confidences, 25)),
                "p50": float(np.percentile(self.confidences, 50)),
                "p75": float(np.percentile(self.confidences, 75)),
                "p95": float(np.percentile(self.confidences, 95))
            }
        
        # Calculate per-class metrics
        if self.predictions and self.true_labels:
            unique_labels = sorted(set(self.true_labels))
            
            for label in unique_labels:
                # Get indices for this class
                true_indices = [i for i, l in enumerate(self.true_labels) if l == label]
                pred_indices = [i for i, l in enumerate(self.predictions) if l == label]
                
                # True positives, false positives, false negatives
                tp = len([i for i in true_indices if self.predictions[i] == label])
                fp = len([i for i in pred_indices if self.true_labels[i] != label])
                fn = len([i for i in true_indices if self.predictions[i] != label])
                
                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                self.per_class_metrics[label] = {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "support": len(true_indices),
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn
                }
    
    def to_dict(self) -> Dict[str, Any]:
        """Enhanced dictionary representation"""
        return {
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'inference_time_ms': self.inference_time_ms,
            'sample_count': len(self.predictions),
            'failed_predictions': self.failed_predictions,
            'success_rate': (len(self.predictions) - self.failed_predictions) / len(self.predictions) * 100 if self.predictions else 0,
            'confidence_distribution': self.confidence_distribution,
            'per_class_metrics': self.per_class_metrics
        }
    
    def meets_quality_threshold(self, min_accuracy: float, min_f1: float) -> Tuple[bool, List[str]]:
        """Check if results meet minimum quality thresholds"""
        issues = []
        
        if self.accuracy < min_accuracy:
            issues.append(f"Accuracy {self.accuracy:.3f} below threshold {min_accuracy:.3f}")
        
        if self.f1_score < min_f1:
            issues.append(f"F1-score {self.f1_score:.3f} below threshold {min_f1:.3f}")
        
        if self.failed_predictions > 0:
            failure_rate = self.failed_predictions / len(self.predictions) * 100
            if failure_rate > 5:  # More than 5% failures
                issues.append(f"High failure rate: {failure_rate:.1f}%")
        
        if self.confidence_distribution and self.confidence_distribution.get("mean", 0) < 0.5:
            issues.append(f"Low average confidence: {self.confidence_distribution['mean']:.3f}")
        
        return len(issues) == 0, issues