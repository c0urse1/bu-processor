import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Dict
from .config import TrainConfig

def load_dataset(cfg: TrainConfig) -> DatasetDict:
    train_df = pd.read_csv(cfg.train_path)
    val_df = pd.read_csv(cfg.val_path)
    data = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False)
    })
    return data

def encode_labels(ds: DatasetDict, cfg: TrainConfig) -> DatasetDict:
    label2id: Dict[str, int] = {l: i for i, l in enumerate(cfg.label_list)}
    id2label: Dict[int, str] = {i: l for l, i in label2id.items()}

    def map_labels(example):
        example["labels"] = label2id[str(example[cfg.label_col])]
        return example

    ds = ds.map(map_labels)
    ds = ds.remove_columns([cfg.label_col])  # behalten: text, labels
    ds = ds.cast_column("labels", ds["train"].features["labels"].__class__ or None)
    ds = ds.with_format("torch")
    return ds, label2id, id2label

from transformers import AutoModelForSequenceClassification

def build_model(cfg: TrainConfig, id2label):
    num_labels = len(cfg.label_list)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()}
    )
    return model
