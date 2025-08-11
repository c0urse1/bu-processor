from bu_processor.training.config import TrainConfig
from bu_processor.training.train import train

def test_train_runs(tmp_path, monkeypatch):
    cfg = TrainConfig(
        train_path="data/train.csv",
        val_path="data/val.csv",
        output_dir=str(tmp_path / "artifacts"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        fp16=False
    )
    out_dir = train(cfg)
    assert (tmp_path / "artifacts").exists()
