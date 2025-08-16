from bu_processor.training.config import TrainingConfig
from bu_processor.training.train import train

def test_train_runs(tmp_path, dummy_train_val):
    """Training Smoke Test mit Dummy-CSV Dateien.
    
    Verwendet dummy_train_val fixture um temporäre CSV-Dateien 
    mit korrekten Labels zu erstellen, statt echte data/train.csv zu benötigen.
    """
    train_path, val_path = dummy_train_val
    
    cfg = TrainingConfig(
        train_path=train_path,
        val_path=val_path,
        output_dir=str(tmp_path / "artifacts"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        fp16=False,
        # Reduziere für Smoke Test
        max_length=128,
        learning_rate=5e-5
    )
    
    out_dir = train(cfg)
    assert (tmp_path / "artifacts").exists()
