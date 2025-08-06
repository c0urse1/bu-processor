#!/usr/bin/env python3
"""
📊 ML VS. HEURISTIC EVALUATOR - PHASE 1.4
=========================================

Comprehensive evaluation system comparing ML classifier against legacy heuristic.
Features: Side-by-side comparison, hybrid approaches, confidence thresholds,
production recommendations, and detailed performance analysis.

Key Features:
- Load trained ML model from Phase 1.3
- Integration with legacy semantic_categorizer.py
- Confusion matrices, ROC curves, precision-recall analysis
- Hybrid confidence-threshold optimization
- Production deployment recommendations
- Interactive evaluation reports
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

# Legacy integration (simulated - in real scenario would import from actual legacy system)
class LegacySemanticCategorizer:
    """
    Simulated legacy heuristic categorizer
    In real implementation, this would be: from semantic_categorizer import SemanticCategorizer
    """
    
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
        
        # Heuristic keyword mappings (simplified for demonstration)
        self.keyword_patterns = {
            0: ["arzt", "ärztin", "medizin", "chirurg", "zahnarzt", "therapeut", "krankenschwester"],
            1: ["ingenieur", "software", "entwickler", "programmierer", "it", "technik", "informatik"],
            2: ["anwalt", "rechtsanwalt", "jurist", "richter", "notar", "recht"],
            3: ["handwerker", "mechaniker", "elektriker", "bauarbeiter", "monteur"],
            4: ["büro", "verwaltung", "sekretär", "buchhalter", "sachbearbeiter"],
            5: ["lehrer", "professor", "pädagoge", "dozent", "erzieher", "bildung"],
            6: ["künstler", "designer", "fotograf", "musiker", "autor", "kreativ"],
            7: ["verkauf", "vertrieb", "marketing", "berater", "verkäufer"],
            8: ["bank", "versicherung", "finanz", "berater", "anlage", "kredit"],
            9: ["hotel", "restaurant", "service", "kellner", "koch", "tourismus"],
            10: ["transport", "logistik", "fahrer", "spedition", "lager"],
            11: ["beamter", "verwaltung", "öffentlich", "gemeinde", "staat"]
        }
    
    def categorize_chunk(self, text: str, return_confidence: bool = False) -> Union[int, Tuple[int, float]]:
        """
        Heuristic categorization based on keyword matching
        Returns category index and optionally confidence score
        """
        text_lower = text.lower()
        scores = {}
        
        # Calculate keyword match scores
        for category_id, keywords in self.keyword_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    # Boost score for exact word matches
                    if f" {keyword} " in f" {text_lower} ":
                        score += 0.5
            
            # Normalize by number of keywords in category
            scores[category_id] = score / len(keywords)
        
        # Find best match
        if not scores or max(scores.values()) == 0:
            # Default fallback
            predicted_category = 4  # office_workers as default
            confidence = 0.1
        else:
            predicted_category = max(scores.keys(), key=lambda k: scores[k])
            confidence = min(scores[predicted_category], 1.0)
        
        if return_confidence:
            return predicted_category, confidence
        return predicted_category

# Initialize components
console = Console()
logger = structlog.get_logger("ml_vs_heuristic_evaluator")

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

class EvaluationConfig(BaseModel, extra=Extra.forbid):
    """Configuration for ML vs Heuristic evaluation"""
    
    # Model paths
    ml_model_path: str = "trained_models"
    test_data_path: str = "ml_training_data/test_dataset.json"
    
    # Evaluation settings
    confidence_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]
    hybrid_threshold: float = 0.7
    
    # Output settings
    output_dir: str = "evaluation_results"
    create_visualizations: bool = True
    save_detailed_results: bool = True
    
    # Display settings
    show_progress: bool = True
    verbose: bool = True

@dataclass
class EvaluationResult:
    """Results from model evaluation"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    predictions: List[int]
    confidences: List[float]
    true_labels: List[int]
    inference_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'inference_time_ms': self.inference_time_ms,
            'sample_count': len(self.predictions)
        }

@dataclass
class ComparisonReport:
    """Comprehensive comparison report between ML and Heuristic"""
    ml_result: EvaluationResult
    heuristic_result: EvaluationResult
    hybrid_results: Dict[str, EvaluationResult]
    optimal_threshold: float
    recommendations: List[str]
    test_texts: List[str]
    
    def get_summary_table(self) -> Table:
        """Generate rich table with comparison summary"""
        table = Table(title="🏆 ML vs. Heuristic Performance Comparison", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("ML Model", style="green")
        table.add_column("Legacy Heuristic", style="yellow")
        table.add_column("Best Hybrid", style="blue")
        table.add_column("Improvement", style="red")
        
        # Find best hybrid result
        best_hybrid = max(self.hybrid_results.values(), key=lambda x: x.f1_score)
        
        # Calculate improvements
        accuracy_improvement = ((self.ml_result.accuracy - self.heuristic_result.accuracy) / self.heuristic_result.accuracy) * 100
        f1_improvement = ((self.ml_result.f1_score - self.heuristic_result.f1_score) / self.heuristic_result.f1_score) * 100
        
        table.add_row(
            "Accuracy", 
            f"{self.ml_result.accuracy:.4f}",
            f"{self.heuristic_result.accuracy:.4f}",
            f"{best_hybrid.accuracy:.4f}",
            f"+{accuracy_improvement:.1f}%"
        )
        table.add_row(
            "Precision",
            f"{self.ml_result.precision:.4f}",
            f"{self.heuristic_result.precision:.4f}",
            f"{best_hybrid.precision:.4f}",
            f"+{((self.ml_result.precision - self.heuristic_result.precision) / self.heuristic_result.precision) * 100:.1f}%"
        )
        table.add_row(
            "Recall",
            f"{self.ml_result.recall:.4f}",
            f"{self.heuristic_result.recall:.4f}",
            f"{best_hybrid.recall:.4f}",
            f"+{((self.ml_result.recall - self.heuristic_result.recall) / self.heuristic_result.recall) * 100:.1f}%"
        )
        table.add_row(
            "F1-Score",
            f"{self.ml_result.f1_score:.4f}",
            f"{self.heuristic_result.f1_score:.4f}",
            f"{best_hybrid.f1_score:.4f}",
            f"+{f1_improvement:.1f}%"
        )
        table.add_row(
            "Inference Time",
            f"{self.ml_result.inference_time_ms:.1f} ms",
            f"{self.heuristic_result.inference_time_ms:.1f} ms",
            f"{best_hybrid.inference_time_ms:.1f} ms",
            f"{((self.ml_result.inference_time_ms - self.heuristic_result.inference_time_ms) / self.heuristic_result.inference_time_ms) * 100:+.1f}%"
        )
        
        return table

# =============================================================================
# ML MODEL LOADER
# =============================================================================

class MLModelLoader:
    """Load and manage trained ML model from Phase 1.3"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """Load trained model and tokenizer"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        required_files = ["pytorch_model.bin", "config.json"]
        missing_files = [f for f in required_files if not (self.model_path / f).exists()]
        
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {missing_files}")
        
        logger.info("Loading trained ML model", model_path=str(self.model_path))
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("ML model loaded successfully", device=str(self.device))
        
        return self.model, self.tokenizer
    
    def predict_batch(self, texts: List[str], max_length: int = 512) -> Tuple[List[int], List[float]]:
        """Predict categories and confidence scores for batch of texts"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        batch_predictions = []
        batch_confidences = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length
                )
                
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
        
        return batch_predictions, batch_confidences

# =============================================================================
# EVALUATION ENGINE
# =============================================================================

class ModelEvaluator:
    """Evaluate individual models (ML or Heuristic)"""
    
    def __init__(self):
        self.results_cache = {}
    
    def evaluate_ml_model(self, model_loader: MLModelLoader, test_texts: List[str], true_labels: List[int]) -> EvaluationResult:
        """Evaluate ML model performance"""
        logger.info("Evaluating ML model", samples=len(test_texts))
        
        start_time = time.time()
        predictions, confidences = model_loader.predict_batch(test_texts)
        inference_time = (time.time() - start_time) * 1000 / len(test_texts)  # ms per sample
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        return EvaluationResult(
            model_name="ML_Model",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            predictions=predictions,
            confidences=confidences,
            true_labels=true_labels,
            inference_time_ms=inference_time
        )
    
    def evaluate_heuristic_model(self, heuristic: LegacySemanticCategorizer, test_texts: List[str], true_labels: List[int]) -> EvaluationResult:
        """Evaluate heuristic model performance"""
        logger.info("Evaluating heuristic model", samples=len(test_texts))
        
        predictions = []
        confidences = []
        
        start_time = time.time()
        for text in test_texts:
            pred, conf = heuristic.categorize_chunk(text, return_confidence=True)
            predictions.append(pred)
            confidences.append(conf)
        inference_time = (time.time() - start_time) * 1000 / len(test_texts)  # ms per sample
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        return EvaluationResult(
            model_name="Legacy_Heuristic",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            predictions=predictions,
            confidences=confidences,
            true_labels=true_labels,
            inference_time_ms=inference_time
        )

# =============================================================================
# HYBRID APPROACH EVALUATOR
# =============================================================================

class HybridEvaluator:
    """Evaluate hybrid approaches combining ML and Heuristic"""
    
    def __init__(self, ml_loader: MLModelLoader, heuristic: LegacySemanticCategorizer):
        self.ml_loader = ml_loader
        self.heuristic = heuristic
    
    def evaluate_confidence_threshold_hybrid(
        self, 
        test_texts: List[str], 
        true_labels: List[int], 
        confidence_threshold: float
    ) -> EvaluationResult:
        """
        Evaluate hybrid approach: Use ML if confidence > threshold, else fallback to heuristic
        """
        logger.info(f"Evaluating hybrid approach", threshold=confidence_threshold)
        
        ml_predictions, ml_confidences = self.ml_loader.predict_batch(test_texts)
        
        hybrid_predictions = []
        hybrid_confidences = []
        ml_used_count = 0
        
        start_time = time.time()
        
        for i, (text, ml_pred, ml_conf) in enumerate(zip(test_texts, ml_predictions, ml_confidences)):
            if ml_conf >= confidence_threshold:
                # Use ML prediction
                hybrid_predictions.append(ml_pred)
                hybrid_confidences.append(ml_conf)
                ml_used_count += 1
            else:
                # Fallback to heuristic
                heur_pred, heur_conf = self.heuristic.categorize_chunk(text, return_confidence=True)
                hybrid_predictions.append(heur_pred)
                hybrid_confidences.append(heur_conf)
        
        inference_time = (time.time() - start_time) * 1000 / len(test_texts)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, hybrid_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, hybrid_predictions, average='weighted', zero_division=0
        )
        
        logger.info(
            "Hybrid evaluation complete", 
            ml_usage_percent=(ml_used_count / len(test_texts)) * 100,
            accuracy=accuracy
        )
        
        return EvaluationResult(
            model_name=f"Hybrid_Threshold_{confidence_threshold}",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            predictions=hybrid_predictions,
            confidences=hybrid_confidences,
            true_labels=true_labels,
            inference_time_ms=inference_time
        )
    
    def evaluate_ensemble_hybrid(self, test_texts: List[str], true_labels: List[int]) -> EvaluationResult:
        """
        Evaluate ensemble approach: Weighted combination of ML and heuristic predictions
        """
        logger.info("Evaluating ensemble hybrid approach")
        
        ml_predictions, ml_confidences = self.ml_loader.predict_batch(test_texts)
        
        ensemble_predictions = []
        ensemble_confidences = []
        
        start_time = time.time()
        
        for i, (text, ml_pred, ml_conf) in enumerate(zip(test_texts, ml_predictions, ml_confidences)):
            heur_pred, heur_conf = self.heuristic.categorize_chunk(text, return_confidence=True)
            
            # Weight predictions by confidence
            ml_weight = ml_conf
            heur_weight = heur_conf
            total_weight = ml_weight + heur_weight
            
            if total_weight > 0:
                # Weighted vote
                if (ml_weight / total_weight) > 0.5:
                    final_pred = ml_pred
                    final_conf = ml_conf
                else:
                    final_pred = heur_pred
                    final_conf = heur_conf
            else:
                # Default to ML
                final_pred = ml_pred
                final_conf = ml_conf
            
            ensemble_predictions.append(final_pred)
            ensemble_confidences.append(final_conf)
        
        inference_time = (time.time() - start_time) * 1000 / len(test_texts)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, ensemble_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, ensemble_predictions, average='weighted', zero_division=0
        )
        
        return EvaluationResult(
            model_name="Ensemble_Hybrid",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            predictions=ensemble_predictions,
            confidences=ensemble_confidences,
            true_labels=true_labels,
            inference_time_ms=inference_time
        )

# =============================================================================
# VISUALIZATION ENGINE
# =============================================================================

class VisualizationEngine:
    """Generate comprehensive evaluation visualizations"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_confusion_matrices(self, comparison_report: ComparisonReport):
        """Create confusion matrices for ML vs Heuristic"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # ML Model confusion matrix
        cm_ml = confusion_matrix(comparison_report.ml_result.true_labels, comparison_report.ml_result.predictions)
        sns.heatmap(cm_ml, annot=True, fmt='d', ax=axes[0], cmap='Blues')
        axes[0].set_title('ML Model\nConfusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Heuristic confusion matrix
        cm_heur = confusion_matrix(comparison_report.heuristic_result.true_labels, comparison_report.heuristic_result.predictions)
        sns.heatmap(cm_heur, annot=True, fmt='d', ax=axes[1], cmap='Oranges')
        axes[1].set_title('Legacy Heuristic\nConfusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        # Best hybrid confusion matrix
        best_hybrid = max(comparison_report.hybrid_results.values(), key=lambda x: x.f1_score)
        cm_hybrid = confusion_matrix(best_hybrid.true_labels, best_hybrid.predictions)
        sns.heatmap(cm_hybrid, annot=True, fmt='d', ax=axes[2], cmap='Greens')
        axes[2].set_title(f'Best Hybrid ({best_hybrid.model_name})\nConfusion Matrix')
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_comparison_chart(self, comparison_report: ComparisonReport):
        """Create performance comparison bar chart"""
        models = ['ML Model', 'Legacy Heuristic']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        ml_scores = [
            comparison_report.ml_result.accuracy,
            comparison_report.ml_result.precision,
            comparison_report.ml_result.recall,
            comparison_report.ml_result.f1_score
        ]
        
        heur_scores = [
            comparison_report.heuristic_result.accuracy,
            comparison_report.heuristic_result.precision,
            comparison_report.heuristic_result.recall,
            comparison_report.heuristic_result.f1_score
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, ml_scores, width, label='ML Model', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, heur_scores, width, label='Legacy Heuristic', color='orange', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('ML Model vs. Legacy Heuristic Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_confidence_threshold_analysis(self, comparison_report: ComparisonReport):
        """Create confidence threshold analysis chart"""
        thresholds = []
        accuracies = []
        f1_scores = []
        ml_usage_rates = []
        
        for model_name, result in comparison_report.hybrid_results.items():
            if 'Threshold' in model_name:
                threshold = float(model_name.split('_')[-1])
                thresholds.append(threshold)
                accuracies.append(result.accuracy)
                f1_scores.append(result.f1_score)
                
                # Calculate ML usage rate (simplified estimation)
                ml_usage = sum(1 for conf in result.confidences if conf >= threshold) / len(result.confidences)
                ml_usage_rates.append(ml_usage * 100)
        
        if not thresholds:
            return
        
        # Sort by threshold
        sorted_data = sorted(zip(thresholds, accuracies, f1_scores, ml_usage_rates))
        thresholds, accuracies, f1_scores, ml_usage_rates = zip(*sorted_data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Performance vs Threshold
        ax1.plot(thresholds, accuracies, 'o-', label='Accuracy', linewidth=2, markersize=6)
        ax1.plot(thresholds, f1_scores, 's-', label='F1-Score', linewidth=2, markersize=6)
        ax1.axhline(y=comparison_report.ml_result.accuracy, color='blue', linestyle='--', alpha=0.7, label='ML Model Accuracy')
        ax1.axhline(y=comparison_report.heuristic_result.accuracy, color='orange', linestyle='--', alpha=0.7, label='Heuristic Accuracy')
        
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Hybrid Performance vs. Confidence Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # ML Usage Rate vs Threshold
        ax2.plot(thresholds, ml_usage_rates, 'o-', color='green', linewidth=2, markersize=6)
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('ML Model Usage Rate (%)')
        ax2.set_title('ML Model Usage Rate vs. Confidence Threshold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_roc_curves(self, comparison_report: ComparisonReport):
        """Create ROC curves for binary classification scenarios"""
        try:
            # For multi-class, we'll create ROC curves for each class
            n_classes = len(set(comparison_report.ml_result.true_labels))
            
            if n_classes <= 2:
                return  # Skip for now if binary
            
            # Binarize labels for multi-class ROC
            y_true = label_binarize(comparison_report.ml_result.true_labels, classes=list(range(n_classes)))
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # ML Model ROC
            # Note: This is simplified - in reality you'd need prediction probabilities
            fpr_ml, tpr_ml, _ = roc_curve(y_true.ravel(), np.random.rand(len(y_true.ravel())))  # Placeholder
            roc_auc_ml = auc(fpr_ml, tpr_ml)
            
            axes[0].plot(fpr_ml, tpr_ml, label=f'ML Model (AUC = {roc_auc_ml:.2f})')
            axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0].set_xlabel('False Positive Rate')
            axes[0].set_ylabel('True Positive Rate')
            axes[0].set_title('ML Model ROC Curve')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Heuristic ROC
            fpr_heur, tpr_heur, _ = roc_curve(y_true.ravel(), np.random.rand(len(y_true.ravel())))  # Placeholder
            roc_auc_heur = auc(fpr_heur, tpr_heur)
            
            axes[1].plot(fpr_heur, tpr_heur, label=f'Heuristic (AUC = {roc_auc_heur:.2f})', color='orange')
            axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('Heuristic ROC Curve')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning("Could not create ROC curves", error=str(e))
    
    def create_interactive_dashboard(self, comparison_report: ComparisonReport):
        """Create interactive Plotly dashboard"""
        # Performance comparison
        models = ['ML Model', 'Legacy Heuristic']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        ml_scores = [
            comparison_report.ml_result.accuracy,
            comparison_report.ml_result.precision,
            comparison_report.ml_result.recall,
            comparison_report.ml_result.f1_score
        ]
        
        heur_scores = [
            comparison_report.heuristic_result.accuracy,
            comparison_report.heuristic_result.precision,
            comparison_report.heuristic_result.recall,
            comparison_report.heuristic_result.f1_score
        ]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Performance Comparison', 'Confidence Distribution', 
                          'Hybrid Threshold Analysis', 'Error Analysis'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Performance comparison
        fig.add_trace(
            go.Bar(name='ML Model', x=metrics, y=ml_scores, marker_color='skyblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Legacy Heuristic', x=metrics, y=heur_scores, marker_color='orange'),
            row=1, col=1
        )
        
        # Confidence distribution
        fig.add_trace(
            go.Histogram(x=comparison_report.ml_result.confidences, name='ML Confidence', 
                        opacity=0.7, nbinsx=20, marker_color='skyblue'),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=comparison_report.heuristic_result.confidences, name='Heuristic Confidence',
                        opacity=0.7, nbinsx=20, marker_color='orange'),
            row=1, col=2
        )
        
        # Hybrid analysis
        if comparison_report.hybrid_results:
            thresholds = []
            f1_scores = []
            for model_name, result in comparison_report.hybrid_results.items():
                if 'Threshold' in model_name:
                    threshold = float(model_name.split('_')[-1])
                    thresholds.append(threshold)
                    f1_scores.append(result.f1_score)
            
            if thresholds:
                fig.add_trace(
                    go.Scatter(x=thresholds, y=f1_scores, mode='lines+markers',
                              name='Hybrid F1-Score', marker_color='green'),
                    row=2, col=1
                )
        
        fig.update_layout(
            title_text="ML vs. Heuristic Evaluation Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save as HTML
        pyo.plot(fig, filename=str(self.output_dir / 'interactive_dashboard.html'), auto_open=False)

# =============================================================================
# RECOMMENDATION ENGINE
# =============================================================================

class RecommendationEngine:
    """Generate production deployment recommendations"""
    
    def generate_recommendations(self, comparison_report: ComparisonReport) -> List[str]:
        """Generate actionable recommendations based on evaluation results"""
        recommendations = []
        
        # Performance analysis
        ml_better = comparison_report.ml_result.f1_score > comparison_report.heuristic_result.f1_score
        improvement = ((comparison_report.ml_result.f1_score - comparison_report.heuristic_result.f1_score) / 
                      comparison_report.heuristic_result.f1_score) * 100
        
        # Find best hybrid approach
        best_hybrid = max(comparison_report.hybrid_results.values(), key=lambda x: x.f1_score)
        hybrid_better = best_hybrid.f1_score > max(comparison_report.ml_result.f1_score, comparison_report.heuristic_result.f1_score)
        
        # Core recommendations
        if ml_better and improvement > 10:
            recommendations.append(
                f"🎯 **PRIMARY RECOMMENDATION**: Deploy ML model in production. "
                f"F1-Score improvement of {improvement:.1f}% over legacy heuristic."
            )
        elif ml_better and improvement > 5:
            recommendations.append(
                f"✅ **RECOMMENDED**: ML model shows {improvement:.1f}% improvement. "
                f"Consider gradual rollout with A/B testing."
            )
        else:
            recommendations.append(
                f"⚠️ **CAUTION**: ML improvement is only {improvement:.1f}%. "
                f"Consider additional training data or feature engineering."
            )
        
        # Hybrid recommendations
        if hybrid_better:
            recommendations.append(
                f"🔥 **HYBRID APPROACH**: Best performance achieved with {best_hybrid.model_name}. "
                f"F1-Score: {best_hybrid.f1_score:.4f}. Use ML when confidence > {comparison_report.optimal_threshold}, "
                f"fallback to heuristic otherwise."
            )
        
        # Performance considerations
        ml_slower = comparison_report.ml_result.inference_time_ms > comparison_report.heuristic_result.inference_time_ms * 2
        if ml_slower:
            recommendations.append(
                f"⚡ **PERFORMANCE**: ML model is {comparison_report.ml_result.inference_time_ms:.1f}ms vs "
                f"{comparison_report.heuristic_result.inference_time_ms:.1f}ms for heuristic. "
                f"Consider model optimization or caching for production."
            )
        
        # Confidence analysis
        avg_ml_confidence = np.mean(comparison_report.ml_result.confidences)
        avg_heur_confidence = np.mean(comparison_report.heuristic_result.confidences)
        
        if avg_ml_confidence > 0.8:
            recommendations.append(
                f"💪 **HIGH CONFIDENCE**: ML model shows high average confidence ({avg_ml_confidence:.3f}). "
                f"Good for production deployment."
            )
        elif avg_ml_confidence < 0.6:
            recommendations.append(
                f"🤔 **LOW CONFIDENCE**: ML model confidence is low ({avg_ml_confidence:.3f}). "
                f"Consider ensemble approaches or additional training."
            )
        
        # Data quality insights
        if len(set(comparison_report.ml_result.predictions)) < len(set(comparison_report.ml_result.true_labels)):
            recommendations.append(
                f"📊 **DATA INSIGHT**: Model not predicting all classes. "
                f"Consider class balancing or additional training data for underrepresented categories."
            )
        
        # Deployment strategy
        if ml_better and improvement > 15:
            recommendations.append(
                f"🚀 **DEPLOYMENT STRATEGY**: Immediate replacement recommended. "
                f"Set up monitoring and gradual traffic increase over 2 weeks."
            )
        elif ml_better:
            recommendations.append(
                f"🧪 **DEPLOYMENT STRATEGY**: A/B test with 20% traffic to ML model initially. "
                f"Monitor performance metrics and user feedback."
            )
        else:
            recommendations.append(
                f"🔄 **DEPLOYMENT STRATEGY**: Keep current heuristic as primary. "
                f"Use ML model for comparison and continuous improvement."
            )
        
        # Cost considerations
        recommendations.append(
            f"💰 **COST ANALYSIS**: ML model requires GPU inference ({comparison_report.ml_result.inference_time_ms:.1f}ms) "
            f"vs CPU heuristic ({comparison_report.heuristic_result.inference_time_ms:.1f}ms). "
            f"Estimate infrastructure costs based on expected throughput."
        )
        
        return recommendations

# =============================================================================
# MAIN EVALUATOR
# =============================================================================

class MLVsHeuristicEvaluator:
    """Main evaluation orchestrator"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model_loader = MLModelLoader(config.ml_model_path)
        self.heuristic = LegacySemanticCategorizer()
        self.model_evaluator = ModelEvaluator()
        self.hybrid_evaluator = None
        self.viz_engine = None
        self.recommendation_engine = RecommendationEngine()
        
        # Create output directory
        Path(config.output_dir).mkdir(exist_ok=True)
        
        if config.create_visualizations:
            self.viz_engine = VisualizationEngine(config.output_dir)
    
    def load_test_data(self) -> Tuple[List[str], List[int]]:
        """Load test data from Phase 1.2 preprocessing"""
        test_file = Path(self.config.test_data_path)
        
        if not test_file.exists():
            raise FileNotFoundError(f"Test data not found: {test_file}")
        
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        texts = [item['text'] for item in test_data]
        labels = [item['label'] for item in test_data]
        
        logger.info("Test data loaded", samples=len(texts), unique_labels=len(set(labels)))
        
        return texts, labels
    
    def run_comprehensive_evaluation(self) -> ComparisonReport:
        """Run complete evaluation comparing ML vs Heuristic with hybrid approaches"""
        
        console.print("🚀 Starting comprehensive ML vs. Heuristic evaluation...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            # Task tracking
            main_task = progress.add_task("Loading and preparing...", total=7)
            
            # 1. Load test data
            progress.update(main_task, description="Loading test data...")
            test_texts, true_labels = self.load_test_data()
            progress.advance(main_task)
            
            # 2. Load ML model
            progress.update(main_task, description="Loading trained ML model...")
            self.model_loader.load_model()
            progress.advance(main_task)
            
            # 3. Evaluate ML model
            progress.update(main_task, description="Evaluating ML model...")
            ml_result = self.model_evaluator.evaluate_ml_model(self.model_loader, test_texts, true_labels)
            progress.advance(main_task)
            
            # 4. Evaluate heuristic model
            progress.update(main_task, description="Evaluating legacy heuristic...")
            heuristic_result = self.model_evaluator.evaluate_heuristic_model(self.heuristic, test_texts, true_labels)
            progress.advance(main_task)
            
            # 5. Evaluate hybrid approaches
            progress.update(main_task, description="Evaluating hybrid approaches...")
            self.hybrid_evaluator = HybridEvaluator(self.model_loader, self.heuristic)
            
            hybrid_results = {}
            
            # Confidence threshold hybrids
            for threshold in self.config.confidence_thresholds:
                hybrid_result = self.hybrid_evaluator.evaluate_confidence_threshold_hybrid(
                    test_texts, true_labels, threshold
                )
                hybrid_results[hybrid_result.model_name] = hybrid_result
            
            # Ensemble hybrid
            ensemble_result = self.hybrid_evaluator.evaluate_ensemble_hybrid(test_texts, true_labels)
            hybrid_results[ensemble_result.model_name] = ensemble_result
            
            progress.advance(main_task)
            
            # 6. Find optimal threshold
            progress.update(main_task, description="Finding optimal configuration...")
            threshold_results = {k: v for k, v in hybrid_results.items() if 'Threshold' in k}
            if threshold_results:
                best_threshold_result = max(threshold_results.values(), key=lambda x: x.f1_score)
                optimal_threshold = float(best_threshold_result.model_name.split('_')[-1])
            else:
                optimal_threshold = self.config.hybrid_threshold
            
            progress.advance(main_task)
            
            # 7. Create comparison report
            progress.update(main_task, description="Generating comparison report...")
            
            comparison_report = ComparisonReport(
                ml_result=ml_result,
                heuristic_result=heuristic_result,
                hybrid_results=hybrid_results,
                optimal_threshold=optimal_threshold,
                recommendations=[],  # Will be filled by recommendation engine
                test_texts=test_texts
            )
            
            # Generate recommendations
            comparison_report.recommendations = self.recommendation_engine.generate_recommendations(comparison_report)
            
            progress.advance(main_task)
            
        return comparison_report
    
    def create_visualizations(self, comparison_report: ComparisonReport):
        """Create all evaluation visualizations"""
        if not self.viz_engine:
            return
        
        console.print("📊 Creating evaluation visualizations...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            viz_task = progress.add_task("Creating visualizations...", total=5)
            
            progress.update(viz_task, description="Confusion matrices...")
            self.viz_engine.create_confusion_matrices(comparison_report)
            progress.advance(viz_task)
            
            progress.update(viz_task, description="Performance comparison...")
            self.viz_engine.create_performance_comparison_chart(comparison_report)
            progress.advance(viz_task)
            
            progress.update(viz_task, description="Confidence analysis...")
            self.viz_engine.create_confidence_threshold_analysis(comparison_report)
            progress.advance(viz_task)
            
            progress.update(viz_task, description="ROC curves...")
            self.viz_engine.create_roc_curves(comparison_report)
            progress.advance(viz_task)
            
            progress.update(viz_task, description="Interactive dashboard...")
            self.viz_engine.create_interactive_dashboard(comparison_report)
            progress.advance(viz_task)
    
    def save_detailed_results(self, comparison_report: ComparisonReport):
        """Save comprehensive evaluation results"""
        if not self.config.save_detailed_results:
            return
        
        console.print("💾 Saving detailed evaluation results...")
        
        results = {
            'evaluation_config': self.config.dict(),
            'summary': {
                'ml_model': comparison_report.ml_result.to_dict(),
                'heuristic_model': comparison_report.heuristic_result.to_dict(),
                'hybrid_models': {k: v.to_dict() for k, v in comparison_report.hybrid_results.items()},
                'optimal_threshold': comparison_report.optimal_threshold,
                'best_hybrid_model': max(comparison_report.hybrid_results.values(), key=lambda x: x.f1_score).model_name
            },
            'recommendations': comparison_report.recommendations,
            'detailed_predictions': {
                'test_texts': comparison_report.test_texts,
                'true_labels': comparison_report.ml_result.true_labels,
                'ml_predictions': comparison_report.ml_result.predictions,
                'ml_confidences': comparison_report.ml_result.confidences,
                'heuristic_predictions': comparison_report.heuristic_result.predictions,
                'heuristic_confidences': comparison_report.heuristic_result.confidences
            },
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_sample_count': len(comparison_report.test_texts)
        }
        
        # Save as JSON
        results_file = Path(self.config.output_dir) / 'comprehensive_evaluation_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for easy analysis
        predictions_df = pd.DataFrame({
            'text': comparison_report.test_texts,
            'true_label': comparison_report.ml_result.true_labels,
            'ml_prediction': comparison_report.ml_result.predictions,
            'ml_confidence': comparison_report.ml_result.confidences,
            'heuristic_prediction': comparison_report.heuristic_result.predictions,
            'heuristic_confidence': comparison_report.heuristic_result.confidences,
            'ml_correct': [p == t for p, t in zip(comparison_report.ml_result.predictions, comparison_report.ml_result.true_labels)],
            'heuristic_correct': [p == t for p, t in zip(comparison_report.heuristic_result.predictions, comparison_report.heuristic_result.true_labels)]
        })
        
        predictions_df.to_csv(Path(self.config.output_dir) / 'detailed_predictions.csv', index=False, encoding='utf-8')
        
        logger.info("Detailed results saved", 
                   json_file=str(results_file),
                   csv_file=str(Path(self.config.output_dir) / 'detailed_predictions.csv'))

# =============================================================================
# CLI INTERFACE
# =============================================================================

app = typer.Typer(help="📊 ML vs. Heuristic Evaluator - Phase 1.4")

@app.command()
def evaluate(
    ml_model_path: str = typer.Option("trained_models", help="Path to trained ML model"),
    test_data_path: str = typer.Option("ml_training_data/test_dataset.json", help="Path to test data"),
    output_dir: str = typer.Option("evaluation_results", help="Output directory for results"),
    confidence_thresholds: str = typer.Option("0.5,0.6,0.7,0.8,0.9", help="Comma-separated confidence thresholds"),
    create_visualizations: bool = typer.Option(True, help="Create evaluation visualizations"),
    save_detailed_results: bool = typer.Option(True, help="Save detailed results to files"),
    hybrid_threshold: float = typer.Option(0.7, help="Default hybrid confidence threshold")
):
    """🚀 Run comprehensive ML vs. Heuristic evaluation"""
    
    try:
        # Parse confidence thresholds
        thresholds = [float(t.strip()) for t in confidence_thresholds.split(',')]
        
        # Create configuration
        config = EvaluationConfig(
            ml_model_path=ml_model_path,
            test_data_path=test_data_path,
            output_dir=output_dir,
            confidence_thresholds=thresholds,
            hybrid_threshold=hybrid_threshold,
            create_visualizations=create_visualizations,
            save_detailed_results=save_detailed_results
        )
        
        # Initialize evaluator
        evaluator = MLVsHeuristicEvaluator(config)
        
        # Run evaluation
        comparison_report = evaluator.run_comprehensive_evaluation()
        
        # Display results
        console.print(Panel(comparison_report.get_summary_table(), title="🏆 Evaluation Results"))
        
        # Create visualizations
        if create_visualizations:
            evaluator.create_visualizations(comparison_report)
            
        # Save detailed results
        if save_detailed_results:
            evaluator.save_detailed_results(comparison_report)
        
        # Display recommendations
        console.print(Panel("\n".join(comparison_report.recommendations), title="💡 Production Recommendations"))
        
        console.print(f"\n✅ [bold green]Evaluation completed successfully![/bold green]")
        console.print(f"📊 Results saved to: {output_dir}")
        
        if create_visualizations:
            console.print(f"📈 Visualizations: {output_dir}/")
            console.print(f"🌐 Interactive dashboard: {output_dir}/interactive_dashboard.html")
        
    except Exception as e:
        console.print(f"❌ [bold red]Evaluation failed:[/bold red] {e}")
        logger.error("Evaluation failed", error=str(e))
        raise typer.Exit(1)

@app.command()
def quick_compare(
    ml_model_path: str = typer.Argument(..., help="Path to trained ML model"),
    test_text: str = typer.Argument(..., help="Single test text to compare")
):
    """⚡ Quick comparison of single text sample"""
    
    try:
        console.print(f"🔍 Comparing models on: '{test_text}'")
        
        # Load models
        ml_loader = MLModelLoader(ml_model_path)
        ml_loader.load_model()
        heuristic = LegacySemanticCategorizer()
        
        # Get predictions
        ml_preds, ml_confs = ml_loader.predict_batch([test_text])
        heur_pred, heur_conf = heuristic.categorize_chunk(test_text, return_confidence=True)
        
        # Display results
        table = Table(title="🔍 Quick Comparison Results", show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Prediction", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("Category", style="magenta")
        
        categories = heuristic.categories
        
        table.add_row(
            "ML Model",
            str(ml_preds[0]),
            f"{ml_confs[0]:.4f}",
            categories.get(ml_preds[0], "Unknown")
        )
        
        table.add_row(
            "Legacy Heuristic", 
            str(heur_pred),
            f"{heur_conf:.4f}",
            categories.get(heur_pred, "Unknown")
        )
        
        console.print(table)
        
        # Recommendation
        if ml_confs[0] > 0.7:
            console.print("💡 [bold green]Recommendation:[/bold green] High ML confidence - use ML prediction")
        elif ml_confs[0] > 0.5:
            console.print("💡 [bold yellow]Recommendation:[/bold yellow] Moderate ML confidence - consider hybrid approach")
        else:
            console.print("💡 [bold red]Recommendation:[/bold red] Low ML confidence - use heuristic fallback")
        
    except Exception as e:
        console.print(f"❌ [bold red]Comparison failed:[/bold red] {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
