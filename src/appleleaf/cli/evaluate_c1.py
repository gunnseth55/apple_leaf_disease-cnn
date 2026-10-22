"""Inference-only, paired three-model comparison on frozen PV and PP2020 tests."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
import torchvision
from torch.utils.data import DataLoader

from appleleaf.config import APPLE_CLASSES
from appleleaf.cli.train_field import FieldDataset
from appleleaf.engine import choose_device, predict
from appleleaf.plots import plot_confusion_matrix
from appleleaf.workflow import class_names, load_model


MODELS = {
    "reference": Path("artifacts/efficientnet_b0/apple_efficientnet_b0_checkpoint.pt"),
    "robust": Path("artifacts/efficientnet_b0_robust/apple_efficientnet_b0_robust_checkpoint.pt"),
    "c1": Path("artifacts/efficientnet_b0_field_c1_v1/apple_efficientnet_b0_field_c1_checkpoint.pt"),
}
PRESENT = {"plantvillage": [0, 1, 2, 3], "pp2020": [0, 2, 3]}
EXPECTED = {"plantvillage": {0: 95, 1: 93, 2: 41, 3: 246},
            "pp2020": {0: 89, 2: 93, 3: 70}}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_mapping(frame):
    mapping = dict(zip(APPLE_CLASSES, range(4)))
    require(frame.class_name.map(mapping).eq(frame.label).all(), "Invalid manifest class mapping")


def same_inventory(left, right):
    """Ignore CSV row order and extra augmentation columns, preserve image identities."""
    columns = ["path", "label", "class_name", "image_hash"]
    for frame in (left, right):
        check_mapping(frame)
        require(not frame.path.duplicated().any(), "Duplicate inventory paths")
    require(left[columns].sort_values("path").reset_index(drop=True).equals(
        right[columns].sort_values("path").reset_index(drop=True)), "Frozen inventories differ")


def validate_metadata(metadata, name):
    require(metadata.get("architecture") == "efficientnet_b0" and
            metadata.get("image_size") == 224 and
            metadata.get("preprocessing") == "imagenet_rgb_resize" and
            metadata.get("class_to_index") == dict(zip(APPLE_CLASSES, range(4))),
            f"Incompatible checkpoint: {name}")
    expected = {"mean": [.485, .456, .406], "std": [.229, .224, .225]}
    require(metadata.get("normalization", expected) == expected,
            f"Unexpected normalization: {name}")
    strategy = {"reference": "plantvillage_only_imagenet_transfer",
                "robust": "plantvillage_only_robust_transfer",
                "c1": "plantvillage_pp2020_source_and_within_source_class_balanced_finetuning"}
    require(metadata.get("training_strategy") == strategy[name], f"Wrong model role: {name}")


def validate_images(frames):
    """Decode all split images and recompute recorded and common oriented identities."""
    seen_paths, seen_pixels, counts = {}, {}, {}
    for split, frame in frames.items():
        check_mapping(frame)
        for row in frame.itertuples():
            path = Path(row.path).resolve()
            require(str(path) not in seen_paths, f"Path overlap within/across splits: {path}")
            with Image.open(path) as image:
                image.load()
                raw = image.convert("RGB")
                oriented = ImageOps.exif_transpose(image).convert("RGB")
                raw_hash = hashlib.sha256(str(raw.size).encode() + raw.tobytes()).hexdigest()
                pixel = hashlib.sha256(str(oriented.size).encode() + oriented.tobytes()).hexdigest()
            if split.startswith("pv_"):
                require(raw_hash == row.image_hash, f"PV pixels changed: {path}")
                require(raw_hash == pixel, f"PV EXIF orientation changes original preprocessing: {path}")
            else:
                require(pixel == row.pixel_sha256 and digest(path) == row.sha256,
                        f"PP2020 bytes/pixels changed: {path}")
            require(pixel not in seen_pixels, f"Pixel overlap within/across splits: {path}")
            seen_paths[str(path)], seen_pixels[pixel] = split, split
        counts[split] = {"total": len(frame), "by_label": frame.label.value_counts().to_dict()}
        print(f"Validated {split}: {len(frame)} readable, unchanged, disjoint images", flush=True)
    return counts


def validate_inputs(checkpoints):
    """Fail before test inference on incompatible checkpoints or changed frozen inputs."""
    project = Path(__file__).resolve().parents[3]
    field = project / "data/field_manifests/pp2020_split_v1"
    pv = checkpoints["reference"].parent
    tracked = {str(path.resolve()): digest(path) for path in checkpoints.values()}
    metadata = {}
    for name, path in checkpoints.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        validate_metadata(checkpoint, name)
        metadata[name] = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        del checkpoint
        # Validate the selected checkpoint against completed, validation-only history.
        history_path = path.parent / "training_history.csv"
        history = pd.read_csv(history_path)
        metric = "selection_score" if name == "c1" else "val_macro_f1"
        best = history.loc[history[metric].idxmax()]
        require(int(best.epoch) == metadata[name]["best_epoch"], f"Validation selection mismatch: {name}")
        saved_score = metadata[name]["best_validation_score" if name == "c1" else "best_val_macro_f1"]
        require(np.isclose(best[metric], saved_score, rtol=0, atol=1e-12), f"Validation score mismatch: {name}")
        require(len(history) == (10 if name == "c1" else 15), f"Incomplete training history: {name}")
        for source in [history_path, path.parent / "run_config.json"]:
            tracked[str(source.resolve())] = digest(source)
    require(metadata["c1"]["best_epoch"] == 10, "Expected the selected C1 epoch-10 checkpoint")
    require(metadata["c1"]["initialization_sha256"] == digest(checkpoints["reference"]),
            "C1 reference initialization changed")
    for path, expected in metadata["c1"]["source_manifest_sha256"].items():
        require(digest(path) == expected, f"C1 frozen input changed: {path}")
        tracked[str(Path(path).resolve())] = expected
    subprocess.run([sys.executable, str(project / "scripts/build_pp2020_field_splits.py"),
                    "--project", str(project), "--output-dir", str(field), "--verify"], check=True)
    frames = {}
    for split in ["train", "validation", "test"]:
        source = pv / f"{split}_manifest.csv"
        frames[f"pv_{split}"] = pd.read_csv(source)
        robust_source = checkpoints["robust"].parent / source.name
        same_inventory(frames[f"pv_{split}"], pd.read_csv(robust_source))
        for path in [source, robust_source]:
            tracked[str(path.resolve())] = digest(path)
    for split in ["train", "val", "test"]:
        source = field / f"{split}.csv"
        frames[f"field_{split}"] = pd.read_csv(source)
        require(not frames[f"field_{split}"].group_id.duplicated().any(), "Duplicate PP2020 group")
        tracked[str(source.resolve())] = digest(source)
    c1_dir = checkpoints["c1"].parent
    mixed = pd.read_csv(c1_dir / "train_manifest.csv")
    for source, key in [("plantvillage", "pv_train"), ("pp2020", "field_train")]:
        selected = mixed[mixed.source.eq(source)]
        identity = ["path", "class_name", "label"]
        require(selected[identity].sort_values("path").reset_index(drop=True).equals(
            frames[key][identity].sort_values("path").reset_index(drop=True)), "C1 training inventory mismatch")
    require(len(mixed) == len(frames["pv_train"]) + len(frames["field_train"]), "Unexpected C1 training source")
    for source, key in [("plantvillage", "pv_validation"), ("pp2020", "field_val")]:
        path = c1_dir / f"{source}_validation_manifest.csv"
        saved = pd.read_csv(path)
        identity = ["path", "class_name", "label"]
        require(saved[identity].equals(frames[key][identity]), "C1 validation inventory mismatch")
        tracked[str(path.resolve())] = digest(path)
    tracked[str((c1_dir / "train_manifest.csv").resolve())] = digest(c1_dir / "train_manifest.csv")
    counts = validate_images(frames)
    tests = {"plantvillage": frames["pv_test"], "pp2020": frames["field_test"]}
    for source, frame in tests.items():
        require(frame.label.value_counts().to_dict() == EXPECTED[source], f"Unexpected test counts: {source}")
    return tests, {"passed": True, "input_sha256": tracked, "models": metadata,
                   "all_split_counts": counts,
                   "checks": ["validation-selected checkpoint metadata and histories",
                              "C1 frozen source hashes and reference initialization",
                              "PP2020 full frozen split verification and deterministic replay",
                              "reference/robust identical PV split inventories",
                              "C1 saved training and validation inventories",
                              "all six source splits decoded and pixel/byte identities verified",
                              "no path or normalized-pixel overlap within/across source splits",
                              "exact frozen test class counts; PV EXIF preprocessing parity"]}


def score_predictions(results, present_labels):
    labels, predictions, probabilities = (results[key] for key in ["labels", "predictions", "probabilities"])
    require(probabilities.shape == (len(labels), 4) and np.isfinite(probabilities).all() and
            (probabilities >= 0).all() and (probabilities <= 1).all() and
            np.allclose(probabilities.sum(axis=1), 1, atol=1e-6), "Invalid four-way probabilities")
    require(np.array_equal(predictions, probabilities.argmax(axis=1)), "Prediction/probability mismatch")
    names = class_names()
    report = classification_report(labels, predictions, labels=list(range(4)),
                                   target_names=names, output_dict=True, zero_division=0)
    per_class = {name: {**report[name], "recall": report[name]["recall"] if (labels == i).any() else None}
                 for i, name in enumerate(names)}
    confidence = probabilities.max(axis=1)
    correct = labels == predictions
    bins, ece = [], 0.
    for i in range(10):
        selected = (confidence >= i/10) & ((confidence < (i+1)/10) if i < 9 else (confidence <= 1))
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else None
        mean = float(confidence[selected].mean()) if count else None
        bins.append({"lower": i/10, "upper": (i+1)/10, "count": count,
                     "correct": int((selected & correct).sum()), "errors": int((selected & ~correct).sum()),
                     "accuracy": accuracy, "mean_confidence": mean})
        if count:
            ece += count / len(labels) * abs(accuracy - mean)
    high = confidence >= .90
    summary = {"mean": float(confidence.mean()), "mean_correct": float(confidence[correct].mean()) if correct.any() else None,
               "mean_errors": float(confidence[~correct].mean()) if (~correct).any() else None,
               "high_confidence_threshold": .90, "high_confidence_count": int(high.sum()),
               "high_confidence_errors": int((high & ~correct).sum()),
               "high_confidence_error_rate": float((~correct[high]).mean()) if high.any() else None,
               "ece_10_bins": float(ece),
               "negative_log_likelihood": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-15, 1)).mean()),
               "interpretation": "Uncalibrated softmax; descriptive analysis only, no threshold tuning"}
    return {"sample_count": len(labels), "correct_count": int(correct.sum()),
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=present_labels, average="macro", zero_division=0)),
            "macro_f1_labels": present_labels,
            "macro_f1_all_four": float(f1_score(labels, predictions, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
            "class_order": names, "per_class": per_class,
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2, 3]).tolist(),
            "black_rot_prediction_count": int((predictions == 1).sum()), "confidence": summary, "confidence_bins": bins}


def save_evaluation(output, frame, results, metrics, title):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics.json", metrics)
    pd.DataFrame(metrics["per_class"]).T.rename_axis("class").to_csv(output / "per_class_metrics.csv")
    pd.DataFrame(metrics["confusion_matrix"], index=class_names(), columns=class_names()).rename_axis(
        "true_class").to_csv(output / "confusion_matrix.csv")
    pd.DataFrame(metrics["confidence_bins"]).to_csv(output / "confidence_bins.csv", index=False)
    predictions = frame.copy()
    predictions["prediction"] = results["predictions"]
    predictions["predicted_class"] = [APPLE_CLASSES[i] for i in results["predictions"]]
    predictions["correct"] = results["labels"] == results["predictions"]
    predictions["confidence"] = results["probabilities"].max(axis=1)
    for i, name in enumerate(APPLE_CLASSES):
        predictions[f"probability_{name}"] = results["probabilities"][:, i]
    predictions.to_csv(output / "predictions.csv", index=False)
    predictions[~predictions.correct].sort_values("confidence", ascending=False).to_csv(output / "errors.csv", index=False)
    np.savez_compressed(output / "predictions.npz", **results)
    plot_confusion_matrix(results["labels"], results["predictions"], class_names(), output / "confusion_matrix.png", title=title)
    confidence, correct = predictions.confidence.to_numpy(), predictions.correct.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([confidence[correct], confidence[~correct]], bins=np.linspace(0, 1, 11), stacked=True,
            label=["Correct", "Incorrect"], color=["#4c78a8", "#e45756"])
    ax.set(xlabel="Maximum softmax probability (uncalibrated)", ylabel="Images", title=title, xlim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "confidence_distribution.png", dpi=160)
    plt.close(fig)


def paired_counts(reference, candidate):
    require(np.array_equal(reference["labels"], candidate["labels"]), "Paired labels differ")
    left = reference["predictions"] == reference["labels"]
    right = candidate["predictions"] == candidate["labels"]
    return {"both_correct": int((left & right).sum()), "both_wrong": int((~left & ~right).sum()),
            "corrected": int((~left & right).sum()), "regressed": int((left & ~right).sum())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/c1_heldout_comparison_v1"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true", help="Check inputs without model inference or output writes")
    args = parser.parse_args(argv)
    require(args.batch_size > 0 and args.threads > 0, "Positive batch size and threads required")
    require(not args.output_dir.exists(), "Choose a new output directory; existing runs are preserved")
    output_path = args.output_dir.resolve()
    for protected in [Path("data").resolve(), *(path.parent.resolve() for path in MODELS.values())]:
        require(output_path != protected and protected not in output_path.parents and output_path not in protected.parents,
                "Output must be separate from protected inputs")
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    tests, validation = validate_inputs(MODELS)
    print("INPUT VALIDATION PASSED: no test inference has run.", flush=True)
    if args.validate_only:
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for source, frame in tests.items():
        frame.to_csv(args.output_dir / f"{source}_test_manifest.csv", index=False)
    validation.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                       "script_sha256": digest(__file__), "device": str(device), "batch_size": args.batch_size,
                       "threads": args.threads, "python": platform.python_version(), "torch": str(torch.__version__),
                       "torchvision": str(torchvision.__version__), "training_performed": False,
                       "preprocessing": "EXIF-oriented RGB; full-image 224x224 bicubic; ToTensor; ImageNet mean/std; no augmentation/crop",
                       "test_policy": "Frozen tests; no selection, tuning, renormalization or automatic model promotion",
                       "prior_exposure": "PV test previously scored for reference/robust; held out from training, not first-look evidence"})
    write_json(args.output_dir / "input_validation.json", validation)
    all_results, all_metrics, rows = {}, {}, []
    for name, checkpoint in MODELS.items():
        model, metadata = load_model(checkpoint, device)
        validate_metadata(metadata, name)
        for source, frame in tests.items():
            started = time.perf_counter()
            loader = DataLoader(FieldDataset(frame), batch_size=args.batch_size, shuffle=False, num_workers=0)
            results = predict(model, loader, device)
            require(np.array_equal(results["labels"], frame.label.to_numpy()), "Prediction order differs from frozen manifest")
            metrics = score_predictions(results, PRESENT[source])
            metrics["metadata"] = {"model": name, "source": source, "best_epoch": metadata["best_epoch"],
                                   "checkpoint_sha256": validation["input_sha256"][str(checkpoint.resolve())],
                                   "test_manifest_sha256": digest(args.output_dir / f"{source}_test_manifest.csv"),
                                   "seconds": time.perf_counter() - started}
            save_evaluation(args.output_dir / name / source, frame, results, metrics, f"{name}: {source} frozen test")
            all_results[name, source], all_metrics[name, source] = results, metrics
            rows.append({"model": name, "source": source, "sample_count": len(frame),
                         **{key: metrics[key] for key in ["correct_count", "accuracy", "macro_f1", "macro_f1_all_four", "black_rot_prediction_count"]},
                         **{f"recall_{i}": metrics["per_class"][class_names()[i]]["recall"] for i in range(4)},
                         **{f"confidence_{key}": value for key, value in metrics["confidence"].items() if key != "interpretation"}})
            print(f"{name}/{source}: {metrics['correct_count']}/{len(frame)}; accuracy {metrics['accuracy']:.6f}; "
                  f"macro-F1 {metrics['macro_f1']:.6f}; {metrics['metadata']['seconds']:.1f}s", flush=True)
        del model
    paired = []
    for source in tests:
        for baseline, candidate in [("reference", "robust"), ("reference", "c1"), ("robust", "c1")]:
            paired.append({"source": source, "baseline": baseline, "candidate": candidate,
                           **paired_counts(all_results[baseline, source], all_results[candidate, source]),
                           **{f"delta_{metric}": all_metrics[candidate, source][metric] - all_metrics[baseline, source][metric]
                              for metric in ["accuracy", "macro_f1"]}})
    pd.DataFrame(rows).to_csv(args.output_dir / "comparison.csv", index=False)
    pd.DataFrame(paired).to_csv(args.output_dir / "paired_comparison.csv", index=False)
    # Preserve the exact per-image changes, rather than only aggregate differences.
    for source, frame in tests.items():
        paired_frame = frame[["path", "class_name", "label"]].copy()
        for name in MODELS:
            paired_frame[f"{name}_prediction"] = all_results[name, source]["predictions"]
        paired_frame.to_csv(args.output_dir / f"{source}_paired_predictions.csv", index=False)
    unchanged = all(digest(path) == expected for path, expected in validation["input_sha256"].items())
    require(unchanged, "Frozen input changed during evaluation; do not use this run")
    write_json(args.output_dir / "completion.json", {"complete": True, "all_frozen_inputs_unchanged": unchanged,
               "training_performed": False, "evaluations": 6, "unique_test_images": 727,
               "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    print(f"COMPLETE: six paired evaluations saved in {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
