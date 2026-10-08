"""Re-evaluate frozen classifiers and compare with original saved predictions."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import PIL
import sklearn
import torch
import torchvision
from sklearn.metrics import confusion_matrix

from appleleaf.config import APPLE_CLASSES, ExperimentConfig
from appleleaf.data import build_image_table, find_variant_root
from appleleaf.engine import classification_metrics, predict
from appleleaf.plots import plot_confusion_matrix
from appleleaf.workflow import build_loaders, class_names, load_model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cpu")
    runs = (
        ("baseline", Path("artifacts/apple_baseline_checkpoint.pt"),
         Path("artifacts/test_predictions.npz")),
        ("background_augmented",
         Path("artifacts/background_augmented/apple_background_augmented_checkpoint.pt"),
         Path("artifacts/background_augmented/test_predictions.npz")),
    )
    summary = {
        "dataset_path": str(args.dataset_path.resolve()),
        "device": str(device),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "pillow": PIL.__version__,
                     "scikit_learn": sklearn.__version__, "numpy": np.__version__},
        "split": "sorted class paths; decoded RGB exact deduplication; stratified 70/15/15, seed from checkpoint",
        "preprocessing": "PIL RGB; Resize((128,128)), bilinear; ToTensor; no normalization or augmentation",
        "models": {},
    }
    for name, checkpoint_path, original_path in runs:
        print(f"Evaluating {name}: {checkpoint_path}", flush=True)
        model, checkpoint = load_model(checkpoint_path, device)
        expected_mapping = {name: index for index, name in enumerate(APPLE_CLASSES)}
        if checkpoint["class_to_index"] != expected_mapping:
            raise RuntimeError("Checkpoint class mapping differs from dataset mapping")
        config = ExperimentConfig(image_size=int(checkpoint["image_size"]),
                                  seed=int(checkpoint["seed"]))
        loaders, splits = build_loaders(args.dataset_path, config, verify=True)
        train_df, val_df, test_df = splits
        raw_count = len(build_image_table(find_variant_root(args.dataset_path, "color")))
        output = args.output_dir / name
        output.mkdir()
        for split_name, frame in zip(("train", "validation", "test"), splits):
            frame.to_csv(output / f"{split_name}_manifest.csv", index=False)
        results = predict(model, loaders[2], device)
        original = np.load(original_path)
        same_labels = np.array_equal(results["labels"], original["labels"])
        same_shape = results["predictions"].shape == original["predictions"].shape
        metrics = classification_metrics(results, class_names())
        comparison = {
            "labels_identical": same_labels,
            "predictions_identical": np.array_equal(results["predictions"], original["predictions"]),
            "prediction_disagreements": int(np.count_nonzero(results["predictions"] != original["predictions"])) if same_shape else None,
            "max_probability_absolute_difference": float(np.max(np.abs(results["probabilities"] - original["probabilities"]))) if results["probabilities"].shape == original["probabilities"].shape else None,
            "original_saved_accuracy": float(np.mean(original["labels"] == original["predictions"])),
            "original_predictions_sha256": sha256(original_path),
        }
        metrics.update({
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_test_accuracy": checkpoint["test_accuracy"],
            "checkpoint_test_macro_f1": checkpoint["test_macro_f1"],
            "accuracy_difference": metrics["accuracy"] - checkpoint["test_accuracy"],
            "best_epoch": checkpoint["best_epoch"], "seed": config.seed,
            "image_size": config.image_size, "class_to_index": checkpoint["class_to_index"],
            "raw_images": raw_count, "unique_images": sum(map(len, splits)),
            "split_counts": {key: len(frame) for key, frame in zip(("train", "validation", "test"), splits)},
            "test_class_counts": test_df["class_name"].value_counts().to_dict(),
            "correct": int(np.sum(results["labels"] == results["predictions"])),
            "comparison": comparison,
        })
        np.savez_compressed(output / "test_predictions.npz", **results)
        predictions = test_df.copy()
        predictions["prediction"] = results["predictions"]
        for index, class_name in enumerate(APPLE_CLASSES):
            predictions[f"probability_{class_name}"] = results["probabilities"][:, index]
        predictions.to_csv(output / "per_image_predictions.csv", index=False)
        np.savetxt(output / "confusion_matrix.csv", confusion_matrix(results["labels"], results["predictions"], labels=range(4)), delimiter=",", fmt="%d")
        plot_confusion_matrix(results["labels"], results["predictions"], class_names(), output / "confusion_matrix.png")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary["models"][name] = metrics
        print(f"{name}: {metrics['correct']}/{len(test_df)} = {metrics['accuracy']:.8%}; comparison={comparison}", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
