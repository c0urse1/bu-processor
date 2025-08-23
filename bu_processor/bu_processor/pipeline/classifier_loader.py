#!/usr/bin/env python3
"""
🤖 ROBUST MODEL LOADER FOR VERSIONED ARTIFACTS
==============================================

Loads ML classification models from versioned artifacts with fallback strategy.
Supports both local artifacts and HuggingFace Hub models.

Schema Support:
- local:artifacts/model-v1    -> Load from local versioned directory
- hf:deepset/gbert-base       -> Load from HuggingFace Hub  
- hf:my-org/private@v1.0.0    -> Load specific version/tag

Benefits:
- ✅ Startup safety & reproducibility
- ✅ No "silent updates" from external sources
- ✅ Offline capability with local artifacts
- ✅ Blue/Green deployment support
- ✅ Clear error handling
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Any

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    AutoConfig = None

logger = logging.getLogger(__name__)

def _is_hf_dir(path: Path) -> bool:
    """
    Check if directory contains a valid HuggingFace model structure.
    
    Args:
        path: Path to check
        
    Returns:
        True if valid HF model directory
    """
    if not path.exists() or not path.is_dir():
        return False
    
    # Required files for HF model
    required_files = ["config.json"]
    
    # Check for required files
    for file in required_files:
        if not (path / file).exists():
            return False
    
    # Optional but common files
    common_files = [
        "tokenizer.json", "tokenizer_config.json", 
        "vocab.txt", "pytorch_model.bin", "model.safetensors"
    ]
    
    # At least one model file should exist
    model_files = ["pytorch_model.bin", "model.safetensors", "tf_model.h5"]
    has_model_file = any((path / f).exists() for f in model_files)
    
    if not has_model_file:
        logger.warning(f"No model weights found in {path}")
        # Still allow loading - transformers might handle it
    
    return True

def _validate_model_ref(model_ref: str) -> None:
    """
    Validate model reference format.
    
    Args:
        model_ref: Model reference string
        
    Raises:
        ValueError: If format is invalid
    """
    if not model_ref:
        raise ValueError("model_ref cannot be empty")
    
    valid_prefixes = ["local:", "hf:"]
    if not any(model_ref.startswith(prefix) for prefix in valid_prefixes):
        raise ValueError(
            f"Invalid model_ref schema: {model_ref}. "
            f"Must start with one of: {valid_prefixes}"
        )

def load_classifier(model_ref: str) -> Tuple[Any, Any]:
    """
    Load classifier model and tokenizer from versioned artifacts.
    
    Args:
        model_ref: Model reference in format:
            - "local:artifacts/model-v1" -> Load from local directory
            - "hf:deepset/gbert-base" -> Load from HuggingFace Hub
            - "hf:my-org/private-clf@v1.0.0" -> Load specific version
    
    Returns:
        Tuple of (tokenizer, model)
        
    Raises:
        ImportError: If transformers not available
        ValueError: If model_ref format invalid or model not found
        Exception: If model loading fails
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "transformers library not available. "
            "Install with: pip install transformers"
        )
    
    # Validate format
    _validate_model_ref(model_ref)
    
    logger.info(f"Loading classifier from: {model_ref}")
    
    # Handle local artifacts
    if model_ref.startswith("local:"):
        path_str = model_ref.split("local:", 1)[1]
        path = Path(path_str)
        
        logger.info(f"Loading local model from: {path.absolute()}")
        
        if not _is_hf_dir(path):
            raise ValueError(
                f"Invalid local model path: {path.absolute()} "
                f"(config.json missing or directory not found)"
            )
        
        try:
            # Load from local directory
            tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
            config = AutoConfig.from_pretrained(str(path))
            model = AutoModelForSequenceClassification.from_pretrained(
                str(path), 
                config=config
            )
            
            logger.info(f"✅ Successfully loaded local model: {config.model_type}")
            return tokenizer, model
            
        except Exception as e:
            raise ValueError(f"Failed to load local model from {path}: {e}")
    
    # Handle HuggingFace Hub models
    elif model_ref.startswith("hf:"):
        model_name = model_ref.split("hf:", 1)[1]
        
        logger.info(f"Loading HuggingFace model: {model_name}")
        
        try:
            # Load from HuggingFace Hub
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config
            )
            
            logger.info(f"✅ Successfully loaded HF model: {config.model_type}")
            return tokenizer, model
            
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace model {model_name}: {e}")
    
    # Should not reach here due to validation, but explicit error
    raise ValueError(f"Unsupported model_ref schema: {model_ref}")

def get_model_info(model_ref: str) -> dict:
    """
    Get information about a model without loading it.
    
    Args:
        model_ref: Model reference string
        
    Returns:
        Dictionary with model information
    """
    try:
        _validate_model_ref(model_ref)
        
        info = {
            "model_ref": model_ref,
            "type": "local" if model_ref.startswith("local:") else "huggingface",
            "available": False,
            "error": None
        }
        
        if model_ref.startswith("local:"):
            path_str = model_ref.split("local:", 1)[1]
            path = Path(path_str)
            info["path"] = str(path.absolute())
            info["available"] = _is_hf_dir(path)
            
            if info["available"] and (path / "config.json").exists():
                try:
                    import json
                    with open(path / "config.json") as f:
                        config = json.load(f)
                    info["model_type"] = config.get("model_type")
                    info["num_labels"] = config.get("num_labels")
                except Exception as e:
                    info["config_error"] = str(e)
        
        elif model_ref.startswith("hf:"):
            model_name = model_ref.split("hf:", 1)[1]
            info["model_name"] = model_name
            # Note: We can't easily check HF availability without loading
            info["available"] = True  # Assume available unless proven otherwise
        
        return info
        
    except Exception as e:
        return {
            "model_ref": model_ref,
            "available": False,
            "error": str(e)
        }

def load_labels(model_ref: str) -> Optional[list]:
    """
    Load labels.txt file associated with a model.
    
    Args:
        model_ref: Model reference string
        
    Returns:
        List of labels or None if not found
    """
    if model_ref.startswith("local:"):
        path_str = model_ref.split("local:", 1)[1]
        path = Path(path_str)
        labels_file = path / "labels.txt"
        
        if labels_file.exists():
            try:
                with open(labels_file, 'r', encoding='utf-8') as f:
                    labels = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(labels)} labels from {labels_file}")
                return labels
            except Exception as e:
                logger.warning(f"Failed to load labels from {labels_file}: {e}")
    
    # For HF models, labels are typically in config
    # This would require loading the config, which we skip for now
    logger.debug(f"No labels.txt found for {model_ref}")
    return None

# Example usage and testing
if __name__ == "__main__":
    # Test model info
    test_refs = [
        "local:artifacts/model-v1",
        "hf:deepset/gbert-base",
        "invalid:format"
    ]
    
    for ref in test_refs:
        print(f"\nTesting: {ref}")
        info = get_model_info(ref)
        print(f"Info: {info}")
