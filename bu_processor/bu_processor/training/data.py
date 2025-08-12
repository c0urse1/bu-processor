import os
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Dict
from .config import TrainingConfig

def load_dataset(cfg: TrainingConfig) -> DatasetDict:
    """Load training and validation datasets from CSV files.
    
    Args:
        cfg: Training configuration with data paths
        
    Returns:
        DatasetDict with train and validation splits
        
    Raises:
        RuntimeError: If training data files are missing (with helpful message for tests)
    """
    # Check if training data files exist
    if not os.path.exists(cfg.train_path):
        raise RuntimeError(
            f"Training data missing: {cfg.train_path}. "
            "In tests, provide dummy CSV via dummy_train_val fixture."
        )
    
    if not os.path.exists(cfg.val_path):
        raise RuntimeError(
            f"Validation data missing: {cfg.val_path}. "
            "In tests, provide dummy CSV via dummy_train_val fixture."
        )
    
    train_df = pd.read_csv(cfg.train_path)
    val_df = pd.read_csv(cfg.val_path)
    data = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False)
    })
    return data

def encode_labels(ds: DatasetDict, cfg: TrainingConfig) -> DatasetDict:
    """Encode string labels to integers for training.
    
    Args:
        ds: Dataset with string labels
        cfg: Training configuration with label list
        
    Returns:
        Tuple of (encoded_dataset, label2id_mapping, id2label_mapping)
    """
    label2id: Dict[str, int] = {l: i for i, l in enumerate(cfg.label_list)}
    id2label: Dict[int, str] = {i: l for l, i in label2id.items()}

    def map_labels(example):
        example["labels"] = label2id[str(example[cfg.label_col])]
        return example

    # Map labels to integers
    ds = ds.map(map_labels)
    
    # Remove original label column (keep: text, labels)
    ds = ds.remove_columns([cfg.label_col])
    
    # Convert to torch format for training
    ds = ds.with_format("torch")
    
    return ds, label2id, id2label

from transformers import AutoModelForSequenceClassification

def build_model(cfg: TrainingConfig, id2label):
    num_labels = len(cfg.label_list)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()}
    )
    return model
