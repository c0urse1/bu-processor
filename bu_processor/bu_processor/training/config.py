from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path

# Bestimme Projektroot basierend auf der Position dieser Datei
# config.py liegt in bu_processor/training/, also 2 Ebenen hoch zum Projektroot
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 2 Ebenen hoch: training -> bu_processor -> projektroot (bu_processor)

class TrainingConfig(BaseModel):
    # Daten - robuste Pfade relativ zum Projektroot
    train_path: str = str(PROJECT_ROOT / "data" / "train.csv")
    val_path: str = str(PROJECT_ROOT / "data" / "val.csv")
    test_path: Optional[str] = str(PROJECT_ROOT / "data" / "test.csv")
    text_col: str = "text"
    label_col: str = "label"
    label_list: List[str] = ["BU_ANTRAG", "POLICE", "BEDINGUNGEN", "SONSTIGES"]

    # Modell & Tokenizer
    model_name: str = "deepset/gbert-base"  # deutsches BERT
    max_length: int = 512

    # Training - robuster output_dir Pfad
    output_dir: str = str(PROJECT_ROOT / "artifacts" / "model-v1")
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 3
    warmup_ratio: float = 0.06
    fp16: bool = True
    seed: int = 42
    eval_strategy: str = "epoch"  # or "steps"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
