import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from .config import TrainingConfig
from .metrics import compute_metrics

def evaluate(cfg: TrainingConfig, model_dir: str, test_path: str):
    df = pd.read_csv(test_path)
    ds = Dataset.from_pandas(df)
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    def tokenize(ex):
        return tok(ex[cfg.text_col], truncation=True, max_length=cfg.max_length)

    ds = ds.map(tokenize, batched=True)
    labels = []
    with open(f"{model_dir}/labels.txt", "r", encoding="utf-8") as f:
        label_list = [line.strip() for line in f.readlines()]
    label2id = {l: i for i, l in enumerate(label_list)}

    def to_ids(ex):
        ex["labels"] = label2id[str(ex[cfg.label_col])]
        return ex

    ds = ds.map(to_ids)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    trainer = Trainer(model=model, tokenizer=tok, compute_metrics=compute_metrics)
    return trainer.evaluate(ds)
