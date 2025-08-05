# =============================================================================
# PROJECT STRUCTURE - MODULAR ENTERPRISE ARCHITECTURE
# =============================================================================

"""
project/
├── data/
│   ├── __init__.py
│   ├── dataset.py          # BUClassificationDataset, DataManager
│   └── preprocessing.py    # Data preprocessing utilities
├── models/
│   ├── __init__.py
│   ├── model.py           # Model definitions, compute_loss, compute_metrics
│   └── config.py          # Model configuration classes
├── training/
│   ├── __init__.py
│   ├── trainer.py         # EnhancedTrainer, TrainingLoop, ResumeLogic
│   ├── callbacks.py       # Custom callbacks for metrics, monitoring
│   └── metrics.py         # Prometheus metrics collection
├── api/
│   ├── __init__.py
│   ├── server.py          # FastAPI app, health/readiness endpoints
│   └── models.py          # Pydantic response models
├── tests/
│   ├── __init__.py
│   ├── test_api.py        # FastAPI endpoint tests
│   ├── test_trainer_resume.py  # Resume training tests
│   ├── test_dataset.py    # Dataset tests
│   └── fixtures/          # Test data and fixtures
│       └── sample.pdf
├── cli.py                 # Typer-CLI glue
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── mypy.ini
├── .pre-commit-config.yaml
└── setup.py
"""

# =============================================================================
# data/dataset.py - DATA MANAGEMENT MODULE
# =============================================================================

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, load_dataset
import datasets
import structlog

logger = structlog.get_logger("data.dataset")

class DataManager:
    """Efficient data loading and caching using HuggingFace datasets"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or "./hf_cache"
        self.stats: Optional[Dict[str, Any]] = None
        
    def load_stats(self, stats_path: Path) -> Dict[str, Any]:
        """Load dataset statistics and recommended hyperparameters"""
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
            
        self.stats = json.loads(stats_path.read_text(encoding='utf-8'))
        logger.info("Stats loaded", stats_file=str(stats_path))
        return self.stats
    
    def load(self, split: str, data_dir: str) -> HFDataset:
        """Load dataset split with intelligent caching"""
        data_file = Path(data_dir) / f"{split}_dataset.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_file}")
        
        # Generate cache key based on file hash
        file_hash = self._get_file_hash(data_file)
        cache_key = f"bu_classifier_{split}_{file_hash[:8]}"
        cache_path = Path(self.cache_dir) / cache_key
        
        try:
            # Try to load from cache
            dataset = datasets.load_from_disk(str(cache_path))
            logger.info("Dataset loaded from cache", split=split, cache_key=cache_key)
            
        except FileNotFoundError:
            # Load from JSON and cache
            logger.info("Loading dataset from JSON", split=split, file=str(data_file))
            
            dataset = load_dataset(
                'json', 
                data_files=str(data_file),
                cache_dir=self.cache_dir
            )['train']
            
            # Save to cache
            os.makedirs(self.cache_dir, exist_ok=True)
            dataset.save_to_disk(str(cache_path))
            logger.info("Dataset cached", split=split, cache_key=cache_key)
        
        return dataset
    
    def load_all_splits(self, data_dir: str) -> Tuple[HFDataset, HFDataset, Dict[str, Any]]:
        """Load train, test splits and stats"""
        data_path = Path(data_dir)
        
        # Load datasets
        train_dataset = self.load("train", data_dir)
        test_dataset = self.load("test", data_dir)
        
        # Load stats
        stats_file = data_path / "dataset_stats.json"
        stats = self.load_stats(stats_file)
        
        logger.info(
            "All splits loaded",
            train_samples=len(train_dataset),
            test_samples=len(test_dataset)
        )
        
        return train_dataset, test_dataset, stats
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate MD5 hash of file for caching"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

class BUClassificationDataset(Dataset):
    """Memory-efficient PyTorch dataset with on-the-fly tokenization"""
    
    def __init__(self, hf_dataset: HFDataset, tokenizer, max_length: int = 512, multi_label: bool = False):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.multi_label = multi_label
        self.num_labels = 12  # Configure based on taxonomy
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Tokenize on-the-fly (memory efficient)
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare labels
        if self.multi_label:
            labels = torch.zeros(self.num_labels)
            label_indices = item['label'] if isinstance(item['label'], list) else [item['label']]
            for label_idx in label_indices:
                if 0 <= label_idx < self.num_labels:
                    labels[label_idx] = 1.0
        else:
            labels = torch.tensor(item['label'], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

# =============================================================================
# models/config.py - MODEL CONFIGURATION
# =============================================================================

from pydantic import BaseModel, validator, Extra
from typing import Optional

class ModelConfig(BaseModel, extra=Extra.forbid):
    """Enhanced model configuration with validation"""
    
    # Model configuration
    model_name: str = "dbmdz/bert-base-german-cased"
    max_length: int = 512
    num_labels: int = 12
    
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Early stopping
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.001
    
    # Performance optimization
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4
    fp16: bool = True
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    auto_find_checkpoint: bool = True
    
    # Output configuration
    output_dir: str = "trained_models"
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # Multi-label support
    multi_label: bool = False
    label_smoothing: float = 0.0
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v <= 0 or v > 128:
            raise ValueError('Batch size must be between 1 and 128')
        return v
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        if v <= 0 or v > 1e-2:
            raise ValueError('Learning rate must be between 0 and 1e-2')
        return v
    
    @validator('num_epochs')
    def validate_epochs(cls, v):
        if v <= 0 or v > 100:
            raise ValueError('Number of epochs must be between 1 and 100')
        return v

# =============================================================================
# models/model.py - MODEL DEFINITIONS AND METRICS
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Any, Tuple
import structlog

logger = structlog.get_logger("models.model")

class ModelManager:
    """Model initialization and management"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def initialize_model_and_tokenizer(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """Initialize BERT model and tokenizer"""
        logger.info(
            "Initializing model", 
            model_name=self.config.model_name,
            device=str(self.device),
            num_labels=self.config.num_labels
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            problem_type="multi_label_classification" if self.config.multi_label else "single_label_classification"
        )
        
        # Move to device
        model.to(self.device)
        
        # Calculate model size
        model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # MB
        
        logger.info(
            "Model initialized",
            model_size_mb=f"{model_size:.1f}",
            parameters=sum(p.numel() for p in model.parameters())
        )
        
        return model, tokenizer
    
    def find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        """Find latest checkpoint in output directory"""
        if not self.config.auto_find_checkpoint:
            return self.config.resume_from_checkpoint
            
        output_path = Path(output_dir)
        if not output_path.exists():
            return None
            
        # Look for checkpoint directories
        checkpoints = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('checkpoint')]
        
        if not checkpoints:
            return None
            
        # Sort by step number and return latest
        try:
            latest = max(checkpoints, key=lambda x: int(x.name.split('-')[-1]))
            logger.info("Found latest checkpoint", checkpoint_path=str(latest))
            return str(latest)
        except (ValueError, IndexError):
            logger.warning("Could not parse checkpoint numbers")
            return None

def compute_loss_single_label(model, inputs, return_outputs=False):
    """Compute loss for single-label classification"""
    labels = inputs.get("labels")
    outputs = model(**inputs)
    
    if labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))
    else:
        loss = outputs.loss
    
    return (loss, outputs) if return_outputs else loss

def compute_loss_multi_label(model, inputs, return_outputs=False):
    """Compute loss for multi-label classification"""
    labels = inputs.get("labels")
    outputs = model(**inputs)
    
    loss_fct = nn.BCEWithLogitsLoss()
    loss = loss_fct(outputs.logits, labels.float())
    
    return (loss, outputs) if return_outputs else loss

def compute_metrics_single_label(eval_pred) -> Dict[str, float]:
    """Compute metrics for single-label classification"""
    predictions, labels = eval_pred
    
    try:
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    except Exception as e:
        logger.error("Metrics computation failed", error=str(e))
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

def compute_metrics_multi_label(eval_pred) -> Dict[str, float]:
    """Compute metrics for multi-label classification"""
    predictions, labels = eval_pred
    
    try:
        predictions = torch.sigmoid(torch.tensor(predictions)).numpy() > 0.5
        accuracy = (predictions == labels).all(axis=1).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='micro', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    except Exception as e:
        logger.error("Multi-label metrics computation failed", error=str(e))
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

# =============================================================================
# training/metrics.py - PROMETHEUS METRICS WITH ENHANCED LABELS
# =============================================================================

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY
from typing import Optional
import structlog

logger = structlog.get_logger("training.metrics")

class MetricsCollector:
    """Centralized metrics collection with enhanced labels"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        
        # Training metrics with enhanced labels
        self.training_epochs = Counter(
            'ml_training_epochs_total', 
            'Total training epochs completed',
            ['model_name', 'task_type', 'resume'],
            registry=self.registry
        )
        
        self.epoch_time = Histogram(
            'training_epoch_duration_seconds',
            'Time per epoch',
            ['epoch', 'split', 'model_name'],
            registry=self.registry
        )
        
        self.training_loss = Gauge(
            'ml_training_loss', 
            'Current training loss',
            ['model_name', 'split', 'epoch'],
            registry=self.registry
        )
        
        self.validation_accuracy = Gauge(
            'ml_validation_accuracy', 
            'Current validation accuracy',
            ['model_name', 'epoch'],
            registry=self.registry
        )
        
        self.model_size_mb = Gauge(
            'ml_model_size_mb', 
            'Model size in megabytes',
            ['model_name'],
            registry=self.registry
        )
        
        self.training_samples = Gauge(
            'ml_training_samples', 
            'Number of training samples',
            ['model_name', 'split'],
            registry=self.registry
        )
        
        self.memory_peak_mb = Gauge(
            'ml_memory_peak_mb', 
            'Peak memory usage during training',
            ['model_name', 'phase'],
            registry=self.registry
        )
        
        # Performance metrics
        self.inference_time = Histogram(
            'ml_inference_duration_seconds', 
            'Inference time per sample',
            ['model_name', 'batch_size'],
            registry=self.registry
        )
        
        # Health metrics
        self.training_errors = Counter(
            'ml_training_errors_total',
            'Training errors by type',
            ['error_type', 'model_name'],
            registry=self.registry
        )
        
        # Resume training metrics
        self.resume_attempts = Counter(
            'ml_resume_attempts_total',
            'Resume training attempts',
            ['model_name', 'success'],
            registry=self.registry
        )

# Global metrics instance
metrics = MetricsCollector()

# =============================================================================
# training/callbacks.py - CUSTOM TRAINING CALLBACKS
# =============================================================================

import time
from transformers import TrainerCallback
import structlog

logger = structlog.get_logger("training.callbacks")

class EpochTimerCallback(TrainerCallback):
    """Callback to measure epoch duration"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.start_time = None
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Mark epoch start time"""
        self.start_time = time.time()
        logger.info("Epoch started", epoch=int(state.epoch), model=self.model_name)
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Record epoch duration"""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            epoch = int(state.epoch)
            
            # Record in Prometheus
            metrics.epoch_time.labels(
                epoch=str(epoch),
                split="train",
                model_name=self.model_name
            ).observe(duration)
            
            logger.info(
                "Epoch completed",
                epoch=epoch,
                duration_seconds=duration,
                model=self.model_name
            )

class MetricsCallback(TrainerCallback):
    """Enhanced metrics callback with detailed logging"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Update metrics at epoch end"""
        epoch = int(state.epoch)
        
        metrics.training_epochs.labels(
            model_name=self.model_name,
            task_type="classification",
            resume=str(args.resume_from_checkpoint is not None)
        ).inc()
        
        logger.info(
            "Training metrics updated",
            epoch=epoch,
            global_step=state.global_step,
            model=self.model_name
        )
    
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        """Update metrics when training logs"""
        if logs is None:
            return
            
        epoch = int(state.epoch)
        
        # Update training loss
        if 'train_loss' in logs:
            metrics.training_loss.labels(
                model_name=self.model_name,
                split="train", 
                epoch=str(epoch)
            ).set(logs['train_loss'])
        
        # Update validation metrics
        if 'eval_loss' in logs:
            metrics.training_loss.labels(
                model_name=self.model_name,
                split="eval",
                epoch=str(epoch)
            ).set(logs['eval_loss'])
            
        if 'eval_accuracy' in logs:
            metrics.validation_accuracy.labels(
                model_name=self.model_name,
                epoch=str(epoch)
            ).set(logs['eval_accuracy'])

# =============================================================================
# training/trainer.py - ENHANCED TRAINER WITH RESUME LOGIC
# =============================================================================

import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding
from typing import Optional, Dict, Any
import structlog
from pathlib import Path

from .callbacks import EpochTimerCallback, MetricsCallback
from .metrics import metrics
from models.model import compute_loss_single_label, compute_loss_multi_label, compute_metrics_single_label, compute_metrics_multi_label
from models.config import ModelConfig

logger = structlog.get_logger("training.trainer")

class EnhancedTrainer(Trainer):
    """Enhanced trainer with resume capabilities and error handling"""
    
    def __init__(self, model_name: str, multi_label: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.multi_label = multi_label
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Enhanced loss computation with error handling"""
        try:
            if self.multi_label:
                return compute_loss_multi_label(model, inputs, return_outputs)
            else:
                return compute_loss_single_label(model, inputs, return_outputs)
                
        except Exception as e:
            logger.error("Loss computation failed", error=str(e), model=self.model_name)
            metrics.training_errors.labels(
                error_type="loss_computation",
                model_name=self.model_name
            ).inc()
            raise
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """Enhanced checkpoint saving with validation"""
        try:
            super()._save_checkpoint(model, trial, metrics)
            logger.info(
                "Checkpoint saved",
                step=self.state.global_step,
                epoch=self.state.epoch,
                model=self.model_name
            )
        except Exception as e:
            logger.error("Checkpoint saving failed", error=str(e), model=self.model_name)
            metrics.training_errors.labels(
                error_type="checkpoint_save",
                model_name=self.model_name
            ).inc()
            raise

class TrainingEngine:
    """Main training engine with resume capabilities"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def create_trainer(self, model, tokenizer, train_dataset, eval_dataset) -> EnhancedTrainer:
        """Create enhanced trainer with all callbacks and resume logic"""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            save_total_limit=self.config.save_total_limit,
            report_to=None,
            label_smoothing_factor=self.config.label_smoothing,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Compute metrics function
        compute_metrics = compute_metrics_multi_label if self.config.multi_label else compute_metrics_single_label
        
        # Create trainer
        trainer = EnhancedTrainer(
            model_name=self.config.model_name,
            multi_label=self.config.multi_label,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                ),
                EpochTimerCallback(self.config.model_name),
                MetricsCallback(self.config.model_name)
            ]
        )
        
        return trainer
    
    def train(self, trainer: EnhancedTrainer, resume: bool = False) -> Dict[str, Any]:
        """Execute training with resume support"""
        checkpoint = None
        
        if resume:
            checkpoint = self.find_latest_checkpoint(self.config.output_dir)
            if checkpoint:
                logger.info("Resuming training from checkpoint", checkpoint=checkpoint)
                metrics.resume_attempts.labels(
                    model_name=self.config.model_name,
                    success="true"
                ).inc()
            else:
                logger.warning("Resume requested but no checkpoint found")
                metrics.resume_attempts.labels(
                    model_name=self.config.model_name,
                    success="false"
                ).inc()
        
        # Start training
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        return {
            'train_result': train_result,
            'eval_result': eval_result,
            'resumed_from': checkpoint
        }
    
    def find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        """Find latest checkpoint directory"""
        output_path = Path(output_dir)
        if not output_path.exists():
            return None
            
        checkpoints = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('checkpoint')]
        
        if not checkpoints:
            return None
            
        try:
            latest = max(checkpoints, key=lambda x: int(x.name.split('-')[-1]))
            return str(latest)
        except (ValueError, IndexError):
            return None

# =============================================================================
# api/models.py - PYDANTIC RESPONSE MODELS
# =============================================================================

from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    training_active: bool = False
    error_count: int = 0
    uptime_seconds: float = 0.0
    timestamp: datetime

class ReadinessResponse(BaseModel):
    """Readiness check response model"""
    status: str
    can_accept_training: bool
    memory_available: bool = True
    gpu_available: bool = False
    model_files_exist: bool = False
    timestamp: datetime

class MetricsSummaryResponse(BaseModel):
    """Training metrics summary response model"""
    validation_accuracy: float = 0.0
    training_loss: float = 0.0
    model_size_mb: float = 0.0
    training_samples: int = 0
    memory_peak_mb: float = 0.0
    training_progress: Dict[str, Any]

class TrainingStatusResponse(BaseModel):
    """Training status response model"""
    is_training: bool
    current_epoch: int = 0
    total_epochs: int = 0
    progress_percent: float = 0.0
    estimated_time_remaining_minutes: Optional[float] = None

# =============================================================================
# api/server.py - FASTAPI SERVER WITH OPENAPI
# =============================================================================

import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from pathlib import Path
import structlog

from .models import HealthResponse, ReadinessResponse, MetricsSummaryResponse, TrainingStatusResponse
from training.metrics import metrics

logger = structlog.get_logger("api.server")

# Global state tracking
class AppState:
    def __init__(self):
        self.training_active = False
        self.start_time = time.time()
        self.error_count = 0
        self.current_epoch = 0
        self.total_epochs = 0

app_state = AppState()

app = FastAPI(
    title="BU Classifier Trainer API",
    description="Enterprise ML Training API with monitoring and health checks",
    version="2.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

def check_pinecone() -> bool:
    """Check Pinecone connectivity"""
    # TODO: Implement actual Pinecone ping
    return True

def check_model_files(model_dir: str = "trained_models") -> bool:
    """Check if trained model files exist"""
    model_path = Path(model_dir)
    required_files = ["pytorch_model.bin", "config.json", "tokenizer.json"]
    
    if not model_path.exists():
        return False
        
    return all((model_path / file).exists() for file in required_files)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced Kubernetes liveness probe with detailed status"""
    uptime = time.time() - app_state.start_time
    
    return HealthResponse(
        status="healthy",
        training_active=app_state.training_active,
        error_count=app_state.error_count,
        uptime_seconds=uptime,
        timestamp=datetime.utcnow()
    )

@app.get("/ready", response_model=ReadinessResponse)
async def readiness_check():
    """Enhanced Kubernetes readiness probe with dependency checks"""
    
    # Check various readiness conditions
    memory_ok = True  # Could implement actual memory check
    gpu_ok = torch.cuda.is_available()
    pinecone_ok = check_pinecone()
    models_ok = check_model_files()
    
    # Fail if too many errors
    if app_state.error_count > 10:
        raise HTTPException(
            status_code=503, 
            detail=f"Too many errors ({app_state.error_count}) - not ready"
        )
    
    # Fail if critical dependencies are down
    if not pinecone_ok:
        raise HTTPException(status_code=503, detail="Pinecone not available")
    
    return ReadinessResponse(
        status="ready",
        can_accept_training=not app_state.training_active,
        memory_available=memory_ok,
        gpu_available=gpu_ok,
        model_files_exist=models_ok,
        timestamp=datetime.utcnow()
    )

@app.get("/metrics-summary", response_model=MetricsSummaryResponse)
async def metrics_summary():
    """Detailed training metrics summary with progress tracking"""
    
    progress_percent = 0.0
    if app_state.total_epochs > 0:
        progress_percent = (app_state.current_epoch / app_state.total_epochs) * 100
    
    return MetricsSummaryResponse(
        validation_accuracy=0.0,  # Would get from metrics
        training_loss=0.0,
        model_size_mb=0.0,
        training_samples=0,
        memory_peak_mb=0.0,
        training_progress={
            "current_epoch": app_state.current_epoch,
            "total_epochs": app_state.total_epochs,
            "progress_percent": progress_percent,
            "training_active": app_state.training_active
        }
    )

@app.get("/training-status", response_model=TrainingStatusResponse)
async def training_status():
    """Current training status and progress"""
    
    progress_percent = 0.0
    estimated_time = None
    
    if app_state.total_epochs > 0:
        progress_percent = (app_state.current_epoch / app_state.total_epochs) * 100
        
        # Simple time estimation based on current progress
        if app_state.current_epoch > 0 and app_state.training_active:
            elapsed = time.time() - app_state.start_time
            avg_time_per_epoch = elapsed / app_state.current_epoch
            remaining_epochs = app_state.total_epochs - app_state.current_epoch
            estimated_time = (remaining_epochs * avg_time_per_epoch) / 60  # minutes
    
    return TrainingStatusResponse(
        is_training=app_state.training_active,
        current_epoch=app_state.current_epoch,
        total_epochs=app_state.total_epochs,
        progress_percent=progress_percent,
        estimated_time_remaining_minutes=estimated_time
    )

@app.post("/training/start")
async def start_training():
    """Start training endpoint (placeholder)"""
    if app_state.training_active:
        raise HTTPException(status_code=409, detail="Training already in progress")
    
    # TODO: Implement actual training start logic
    app_state.training_active = True
    logger.info("Training started via API")
    
    return {"status": "training_started", "timestamp": datetime.utcnow()}

@app.post("/training/stop")
async def stop_training():
    """Stop training endpoint (placeholder)"""
    if not app_state.training_active:
        raise HTTPException(status_code=409, detail="No training in progress")
    
    # TODO: Implement actual training stop logic
    app_state.training_active = False
    logger.info("Training stopped via API")
    
    return {"status": "training_stopped", "timestamp": datetime.utcnow()}

# =============================================================================
# cli.py - TYPER CLI GLUE
# =============================================================================

import typer
from rich.console import Console
from pathlib import Path
import uvicorn
from prometheus_client import start_http_server
import threading

from data.dataset import DataManager, BUClassificationDataset
from models.config import ModelConfig
from models.model import ModelManager
from training.trainer import TrainingEngine
from api.server import app as health_app, app_state

console = Console()
app = typer.Typer(help="🤖 BU Classifier Trainer - Modular Enterprise Edition")

@app.command()
def train(
    data_dir: str = typer.Argument(..., help="Directory with preprocessed data"),
    config_file: str = typer.Option(None, help="JSON config file path"),
    resume: bool = typer.Option(False, help="Resume from latest checkpoint"),
    health_endpoints: bool = typer.Option(False, help="Start health check endpoints"),
    model_name: str = typer.Option("dbmdz/bert-base-german-cased", help="Model name"),
    batch_size: int = typer.Option(16, help="Batch size"),
    epochs: int = typer.Option(3, help="Number of epochs"),
    output_dir: str = typer.Option("trained_models", help="Output directory")
):
    """🚀 Train classifier with modular architecture"""
    
    # Start health endpoints if requested
    if health_endpoints:
        start_http_server(8001)
        console.print("📊 Prometheus metrics: http://localhost:8001/metrics")
        
        def run_health_server():
            uvicorn.run(health_app, host="0.0.0.0", port=8002, log_level="warning")
        
        health_thread = threading.Thread(target=run_health_server, daemon=True)
        health_thread.start()
        console.print("🏥 Health endpoints: http://localhost:8002/docs")
    
    try:
        # Create config
        if config_file and Path(config_file).exists():
            # Load from JSON config file
            import json
            with open(config_file) as f:
                config_dict = json.load(f)
            config = ModelConfig(**config_dict)
        else:
            # Create from CLI args
            config = ModelConfig(
                model_name=model_name,
                batch_size=batch_size,
                num_epochs=epochs,
                output_dir=output_dir
            )
        
        # Update app state
        app_state.total_epochs = config.num_epochs
        app_state.training_active = True
        
        # Initialize components
        console.print("🔧 Initializing training components...")
        
        data_manager = DataManager()
        model_manager = ModelManager(config)
        training_engine = TrainingEngine(config)
        
        # Load data
        console.print("📂 Loading data...")
        train_dataset, test_dataset, stats = data_manager.load_all_splits(data_dir)
        
        # Apply auto hyperparameters
        if 'recommended_hyperparams' in stats:
            recommended = stats['recommended_hyperparams']
            config.batch_size = recommended.get('batch_size', config.batch_size)
            config.learning_rate = recommended.get('learning_rate', config.learning_rate)
            console.print(f"🎯 Applied auto-hyperparams: batch_size={config.batch_size}")
        
        # Initialize model
        console.print("🤖 Loading model...")
        model, tokenizer = model_manager.initialize_model_and_tokenizer()
        
        # Prepare datasets
        train_torch_dataset = BUClassificationDataset(train_dataset, tokenizer, config.max_length, config.multi_label)
        test_torch_dataset = BUClassificationDataset(test_dataset, tokenizer, config.max_length, config.multi_label)
        
        # Create trainer
        trainer = training_engine.create_trainer(model, tokenizer, train_torch_dataset, test_torch_dataset)
        
        # Train
        console.print("🚀 Starting training...")
        results = training_engine.train(trainer, resume=resume)
        
        # Update app state
        app_state.training_active = False
        
        console.print("✅ Training completed!")
        console.print(f"📈 Final accuracy: {results['eval_result']['eval_accuracy']:.4f}")
        
        if results['resumed_from']:
            console.print(f"🔄 Resumed from: {results['resumed_from']}")
        
    except Exception as e:
        app_state.training_active = False
        app_state.error_count += 1
        console.print(f"❌ Training failed: {e}")
        raise typer.Exit(1)

@app.command()
def health():
    """🏥 Start health check server"""
    console.print("🏥 Starting health check system...")
    console.print("📊 Prometheus: http://localhost:8001/metrics")
    console.print("🩺 Health API: http://localhost:8002/docs")
    
    # Start Prometheus
    start_http_server(8001)
    
    # Start FastAPI
    uvicorn.run(health_app, host="0.0.0.0", port=8002)

@app.command()
def validate_data(data_dir: str = typer.Argument(..., help="Data directory")):
    """🔍 Validate preprocessed data"""
    try:
        data_manager = DataManager()
        train_dataset, test_dataset, stats = data_manager.load_all_splits(data_dir)
        
        console.print(f"✅ Training samples: {len(train_dataset)}")
        console.print(f"✅ Test samples: {len(test_dataset)}")
        console.print("✅ Data validation successful!")
        
    except Exception as e:
        console.print(f"❌ Validation failed: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
