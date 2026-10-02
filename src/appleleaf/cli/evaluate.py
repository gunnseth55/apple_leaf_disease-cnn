import argparse
from pathlib import Path

import numpy as np

from appleleaf.config import ExperimentConfig
from appleleaf.data import download_dataset
from appleleaf.engine import choose_device, classification_metrics, predict
from appleleaf.plots import plot_confusion_matrix
from appleleaf.workflow import build_loaders, class_names, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved Apple leaf CNN")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    config = ExperimentConfig(
        image_size=int(checkpoint["image_size"]),
        batch_size=args.batch_size,
        seed=int(checkpoint["seed"]),
    )
    dataset_path = args.dataset_path or download_dataset()
    (_, _, test_loader), _ = build_loaders(dataset_path, config)
    results = predict(model, test_loader, device)
    metrics = classification_metrics(results, class_names())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "test_predictions.npz", **results)
    plot_confusion_matrix(
        results["labels"],
        results["predictions"],
        class_names(),
        args.output_dir / "confusion_matrix.png",
    )
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}\n")
    print(metrics["report"])
    print(f"Evaluation saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

