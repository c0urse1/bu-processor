import argparse, json
from .config import TrainingConfig
from .train import train
from .evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(prog="bu-train")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--model_dir", type=str, default="artifacts/model-v1")
    parser.add_argument("--test_path", type=str, default="data/test.csv")
    args = parser.parse_args()

    cfg = TrainingConfig()  # optional: YAML einlesen und mergen

    if args.train:
        out = train(cfg)
        print(f"trained_model_dir={out}")

    if args.eval:
        metrics = evaluate(cfg, args.model_dir, args.test_path)
        print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
