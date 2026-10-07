"""Frozen-checkpoint evaluation on Kashmir folder labels, without retraining."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from appleleaf.datasets import AppleLeafDataset, build_transform
from appleleaf.engine import choose_device, predict
from appleleaf.workflow import load_model


FOLDER_CLASSES = {
    "APPLE ROT LEAVES": "Apple___Black_rot",
    "HEALTHY LEAVES": "Apple___healthy",
    "SCAB LEAVES": "Apple___Apple_scab",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_manifest(root, class_to_index, exclude_label_conflicts=False):
    if set(class_to_index) != set(FOLDER_CLASSES.values()) | {"Apple___Cedar_apple_rust"}:
        raise ValueError("Checkpoint must contain the four expected Apple classes")
    if sorted(class_to_index.values()) != list(range(4)):
        raise ValueError("Checkpoint class indices must be contiguous from zero")
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    unknown = [p.name for p in root.iterdir() if p.is_dir() and p.name not in FOLDER_CLASSES]
    if unknown:
        raise ValueError(f"Unmapped dataset folders: {unknown}")
    rows, rejected = [], []
    for folder, name in FOLDER_CLASSES.items():
        directory = root / folder
        if not directory.is_dir():
            raise ValueError(f"Missing class folder: {directory}")
        count = 0
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                with Image.open(path) as image:
                    image.convert("RGB").load()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, ValueError) as error:
                pointer = path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
                rejected.append({"path": str(path.resolve()), "relative_path": path.relative_to(root).as_posix(),
                                 "reason": "missing_git_lfs_image" if pointer else "unreadable_image", "error": str(error)})
                continue
            rows.append({"path": str(path.resolve()), "relative_path": path.relative_to(root).as_posix(),
                         "source_folder": folder, "class_name": name,
                         "label": class_to_index[name], "sha256": digest})
            count += 1
        if not count:
            raise ValueError(f"No readable images in {directory}")
    frame = pd.DataFrame(rows)
    conflicts = []
    for _, group in frame.groupby("sha256"):
        if group.label.nunique() > 1:
            conflicts.extend(group.index.tolist())
    if conflicts and not exclude_label_conflicts:
        raise ValueError("Identical image bytes have conflicting class labels; use --exclude-label-conflicts to record and exclude every conflicting row")
    excluded = frame.loc[conflicts].copy()
    frame = frame.drop(index=conflicts).reset_index(drop=True)
    if set(frame.label) != set(class_to_index[name] for name in FOLDER_CLASSES.values()):
        raise ValueError("Conflict exclusion leaves a represented class empty")
    return frame, rejected, excluded


def confidence_summary(values):
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "min": None, "median": None,
                "p90": None, "max": None}
    return {"count": len(values), "mean": float(np.mean(values)), "std": float(np.std(values)),
            "min": float(np.min(values)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "max": float(np.max(values))}


def summarize(results, names):
    labels, predictions, probabilities = (results[k] for k in ("labels", "predictions", "probabilities"))
    present = sorted(np.unique(labels).tolist())
    confidence = probabilities.max(axis=1)
    correct = labels == predictions
    bins = []
    for low, high in zip(np.arange(0, 1, .1), np.arange(.1, 1.1, .1)):
        selected = (confidence >= low) & ((confidence < high) if high < .999 else (confidence <= 1))
        bins.append({"lower": float(low), "upper": float(high), "count": int(selected.sum()),
                     "accuracy": float(correct[selected].mean()) if selected.any() else None,
                     "mean_confidence": float(confidence[selected].mean()) if selected.any() else None})
    report = classification_report(labels, predictions, labels=list(range(len(names))),
                                   target_names=names, output_dict=True, zero_division=0)
    # This sensitivity result avoids relying on the provisional rot label.
    nonrot = labels != names.index("Apple___Black_rot")
    high = confidence >= .9
    return {
        "sample_count": len(labels), "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1_present_classes": float(f1_score(labels, predictions, labels=present, average="macro", zero_division=0)),
        "macro_f1_all_four_classes": float(f1_score(labels, predictions, labels=list(range(4)), average="macro", zero_division=0)),
        "present_classes": [names[i] for i in present],
        "absent_classes": [names[i] for i in range(4) if i not in present],
        "per_class": {name: report[name] for name in names},
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(4))).tolist(),
        "confidence": {"all": confidence_summary(confidence), "correct": confidence_summary(confidence[correct]),
                       "incorrect": confidence_summary(confidence[~correct]),
                       "high_confidence_threshold": .9, "high_confidence_count": int(high.sum()),
                       "high_confidence_error_count": int((high & ~correct).sum()),
                       "high_confidence_error_rate": float((~correct)[high].mean()) if high.any() else None,
                       "by_true_class": {name: confidence_summary(confidence[labels == i]) for i, name in enumerate(names)},
                       "bins": bins},
        "excluding_provisional_rot": {"sample_count": int(nonrot.sum()),
                                     "accuracy": float(correct[nonrot].mean())},
    }


def save_confusion(matrix, names, path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(matrix, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
    ax.set_xticks(range(4), names, rotation=25, ha="right")
    ax.set_yticks(range(4), names)
    ax.set(xlabel="Predicted class", ylabel="Folder-derived true class", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_failures(frame, output, count):
    failures = frame.loc[~frame.correct].sort_values(["confidence", "relative_path"], ascending=[False, True])
    failures.to_csv(output / "failures.csv", index=False)
    selected = failures.head(count)
    selected.to_csv(output / "failure_examples.csv", index=False)
    if selected.empty:
        return
    fig, axes = plt.subplots(int(np.ceil(len(selected) / 3)), 3,
                             figsize=(12, 4 * int(np.ceil(len(selected) / 3))), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes.flat, selected.iterrows()):
        with Image.open(row.path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{row.relative_path}\nTrue: {row.true_class}\nPred: {row.predicted_class} ({row.confidence:.1%})", fontsize=8)
    fig.suptitle("Highest-confidence errors (folder-derived labels)")
    fig.tight_layout()
    fig.savefig(output / "failure_examples.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/external_classification/kashmir"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--failure-examples", type=int, default=12)
    parser.add_argument("--rot-label-status", choices=("provisional", "confirmed"), default="provisional")
    parser.add_argument("--exclude-label-conflicts", action="store_true")
    parser.add_argument("--exclude-missing-lfs", action="store_true", help="Record and exclude image files that contain Git LFS pointers")
    args = parser.parse_args()
    if args.batch_size < 1 or args.failure_examples < 0:
        parser.error("batch size must be positive and failure examples nonnegative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty to avoid mixing evaluation runs")
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    names = [name for name, _ in sorted(checkpoint["class_to_index"].items(), key=lambda item: item[1])]
    frame, rejected, excluded = build_manifest(args.dataset_path, checkpoint["class_to_index"], args.exclude_label_conflicts)
    fatal = [row for row in rejected if row["reason"] != "missing_git_lfs_image" or not args.exclude_missing_lfs]
    if fatal:
        raise ValueError(f"{len(fatal)} unreadable images; restore missing LFS content or explicitly use --exclude-missing-lfs")
    loader = DataLoader(AppleLeafDataset(frame, build_transform(int(checkpoint["image_size"]))),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
    results = predict(model, loader, device)
    metrics = summarize(results, names)
    metrics["metadata"] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "dataset_path": str(args.dataset_path.resolve()), "training_strategy": checkpoint.get("training_strategy", "unspecified"),
        "image_size": int(checkpoint["image_size"]), "class_order": names, "folder_mapping": FOLDER_CLASSES,
        "rot_label_status": args.rot_label_status, "duplicate_extra_rows": int(frame.sha256.duplicated().sum()),
        "excluded_conflicting_rows": len(excluded), "excluded_missing_image_rows": len(rejected),
        "source_image_rows": len(frame) + len(excluded) + len(rejected),
        "protocol": "All readable images; frozen weights; original RGB resize + ToTensor; four-way argmax; no tuning.",
        "limitations": ["No cedar-rust ground truth: its recall/F1 are unmeasured; zero-support table entries are placeholders.",
                        "Rot-to-black-rot mapping is provisional unless independently confirmed.",
                        "Softmax confidence is uncalibrated; duplicate counts cover this dataset only, not training overlap."],
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "manifest.csv", index=False)
    excluded.to_csv(output / "excluded_label_conflicts.csv", index=False)
    pd.DataFrame(rejected, columns=["path", "relative_path", "reason", "error"]).to_csv(output / "excluded_missing_images.csv", index=False)
    frame["true_class"] = [names[i] for i in results["labels"]]
    frame["predicted_class"] = [names[i] for i in results["predictions"]]
    frame["confidence"] = results["probabilities"].max(axis=1)
    frame["correct"] = results["labels"] == results["predictions"]
    for i, name in enumerate(names):
        frame[f"probability_{name}"] = results["probabilities"][:, i]
    frame.to_csv(output / "predictions.csv", index=False)
    np.savez_compressed(output / "predictions.npz", **results)
    pd.DataFrame(metrics["per_class"]).T.rename_axis("class").to_csv(output / "per_class_metrics.csv")
    pd.DataFrame(metrics["confusion_matrix"], index=names, columns=names).rename_axis("true_class").to_csv(output / "confusion_matrix.csv")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    text_report = classification_report(results["labels"], results["predictions"], labels=list(range(4)), target_names=names, digits=4, zero_division=0)
    (output / "classification_report.txt").write_text(text_report, encoding="utf-8")
    display = [name.replace("Apple___", "").replace("_", " ") for name in names]
    save_confusion(metrics["confusion_matrix"], display, output / "confusion_matrix.png", f"Kashmir: {metrics['metadata']['training_strategy']}")
    save_failures(frame, output, args.failure_examples)
    print(f"Images: {len(frame)} | Accuracy: {metrics['accuracy']:.4f} | Macro-F1 (present classes): {metrics['macro_f1_present_classes']:.4f}")
    print(text_report)
    print(f"Evaluation saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
