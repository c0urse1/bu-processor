from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from .config import TrainConfig

def build_tokenizer(cfg: TrainConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    return tok

def tokenize_function(example, tok, cfg: TrainConfig):
    return tok(
        example["text"], truncation=True, max_length=cfg.max_length
    )

def build_collator(tok):
    return DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)

def build_model(cfg: TrainConfig, id2label):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()}
    )
    return model
