import os
from transformers import TrainingArguments, Trainer
from .config import TrainingConfig
from .data import load_dataset, encode_labels
from .models import build_tokenizer, tokenize_function, build_collator, build_model
from .metrics import compute_metrics

def train(cfg: TrainingConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    ds = load_dataset(cfg)
    tok = build_tokenizer(cfg)
    tokenized = ds.map(lambda x: tokenize_function(x, tok, cfg), batched=True)
    tokenized, label2id, id2label = encode_labels(tokenized, cfg)

    model = build_model(cfg, id2label)
    collator = build_collator(tok)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        evaluation_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        fp16=cfg.fp16,
        seed=cfg.seed,
        logging_steps=50,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)

    # Label-Mapping persistieren
    with open(os.path.join(cfg.output_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for lbl in cfg.label_list:
            f.write(lbl + "\n")

    return cfg.output_dir
