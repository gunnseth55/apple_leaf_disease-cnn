import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from appleleaf.config import APPLE_CLASSES, ExperimentConfig
from appleleaf.data import download_dataset
from appleleaf.engine import choose_device, classification_metrics, predict
from appleleaf.plots import plot_confusion_matrix
from appleleaf.workflow import build_loaders, class_names, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved Apple classifier on PlantVillage test images")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError("Output directory must be empty; choose a new --output-dir")
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    config = ExperimentConfig(
        image_size=int(checkpoint["image_size"]),
        batch_size=args.batch_size,
        seed=int(checkpoint["seed"]),
    )
    dataset_path = args.dataset_path or download_dataset()
    expected_mapping = {name: i for i, name in enumerate(APPLE_CLASSES)}
    if checkpoint["class_to_index"] != expected_mapping:
        raise ValueError("Checkpoint class mapping does not match the PlantVillage split")
    (_, _, test_loader), (_, _, test_df) = build_loaders(
        dataset_path, config,
        preprocessing=checkpoint.get("preprocessing", "rgb_resize_totensor"),
    )
    results = predict(model, test_loader, device)
    metrics = classification_metrics(results, class_names())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = class_names()
    report = classification_report(results["labels"], results["predictions"],
                                   labels=list(range(4)), target_names=names,
                                   output_dict=True, zero_division=0)
    per_class = pd.DataFrame({name: report[name] for name in names}).T
    per_class.index.name = "class"
    per_class.to_csv(args.output_dir / "per_class_metrics.csv")
    matrix = confusion_matrix(results["labels"], results["predictions"], labels=list(range(4)))
    pd.DataFrame(matrix, index=names, columns=names).rename_axis("true_class").to_csv(
        args.output_dir / "confusion_matrix.csv")
    confidence = results["probabilities"].max(axis=1)
    correct = results["labels"] == results["predictions"]
    bins = []
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        selected = (confidence >= low) & ((confidence < high) if index < 9 else (confidence <= high))
        bins.append({"lower": low, "upper": high, "count": int(selected.sum()),
                     "correct": int((selected & correct).sum()),
                     "errors": int((selected & ~correct).sum()),
                     "accuracy": float(correct[selected].mean()) if selected.any() else None})
    pd.DataFrame(bins).to_csv(args.output_dir / "confidence_bins.csv", index=False)
    frame = test_df.copy()
    frame["predicted_class"] = [names[i] for i in results["predictions"]]
    frame["confidence"] = confidence
    frame["correct"] = correct
    for i, name in enumerate(names):
        frame[f"probability_{name}"] = results["probabilities"][:, i]
    frame.to_csv(args.output_dir / "predictions.csv", index=False)
    metrics.update({"per_class": {name: report[name] for name in names},
                    "confusion_matrix": matrix.tolist(), "class_order": names,
                    "sample_count": len(confidence),
                    "confidence": {"mean": float(confidence.mean()),
                                   "median": float(np.median(confidence)),
                                   "mean_correct": float(confidence[correct].mean()) if correct.any() else None,
                                   "mean_errors": float(confidence[~correct].mean()) if (~correct).any() else None,
                                   "bins": bins},
                    "metadata": {"checkpoint": str(args.checkpoint.resolve()),
                                 "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                 "architecture": checkpoint.get("architecture", "appleleaf_cnn"),
                                 "preprocessing": checkpoint.get("preprocessing", "rgb_resize_totensor"),
                                 "image_size": config.image_size, "seed": config.seed,
                                 "dataset_path": str(dataset_path.resolve()),
                                 "split": "PlantVillage test; no training or tuning"}})
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    (args.output_dir / "classification_report.txt").write_text(metrics["report"], encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 5))
    edges = np.linspace(0, 1, 11)
    ax.hist([confidence[correct], confidence[~correct]], bins=edges,
            label=["Correct", "Incorrect"], stacked=True, color=["#4c78a8", "#e45756"])
    ax.set(xlabel="Maximum softmax probability (uncalibrated)", ylabel="Image count",
           title="PlantVillage test confidence distribution", xlim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "confidence_distribution.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(args.output_dir / "test_predictions.npz", **results)
    plot_confusion_matrix(
        results["labels"],
        results["predictions"],
        class_names(),
        args.output_dir / "confusion_matrix.png",
        title=f"{checkpoint.get('architecture', 'Baseline CNN')} - Test Confusion Matrix",
    )
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}\n")
    print(metrics["report"])
    print(f"Evaluation saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

