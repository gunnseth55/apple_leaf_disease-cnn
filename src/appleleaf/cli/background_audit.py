import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from appleleaf.config import ExperimentConfig
from appleleaf.data import attach_segmented_paths, download_dataset, prepare_splits
from appleleaf.datasets import AUDIT_CONDITIONS, BACKGROUND_STRESS_CONDITIONS, BackgroundTestDataset, checkpoint_transform
from appleleaf.engine import choose_device, predict
from appleleaf.workflow import class_names, load_model
from appleleaf.config import APPLE_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Audit sensitivity to image backgrounds")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/background_audit.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--include-texture", action="store_true",
                        help="Also test a fixed neutral/earth/foliage multiscale texture")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; choose a new --output")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    if checkpoint["class_to_index"] != dict(zip(APPLE_CLASSES, range(4))):
        raise ValueError("Checkpoint class mapping does not match PlantVillage")
    config = ExperimentConfig(
        image_size=int(checkpoint["image_size"]), seed=int(checkpoint["seed"])
    )
    dataset_path = args.dataset_path or download_dataset()
    _, validation_df, test_df = prepare_splits(dataset_path, config.seed)
    val_df = validation_df if args.split == "validation" else test_df
    val_df = attach_segmented_paths(val_df, dataset_path)
    transform = checkpoint_transform(checkpoint)
    results = {}
    conditions = BACKGROUND_STRESS_CONDITIONS if args.include_texture else AUDIT_CONDITIONS
    for background_name in conditions:
        dataset = BackgroundTestDataset(val_df, background_name, transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        results[background_name] = predict(model, loader, device)

    original_predictions = results["original"]["predictions"]
    rows, per_class, details = [], [], {}
    for background_name, result in results.items():
        confidence = result["probabilities"].max(axis=1)
        report = classification_report(result["labels"], result["predictions"],
                                       labels=list(range(4)), target_names=class_names(),
                                       output_dict=True, zero_division=0)
        details[background_name] = {
            "per_class": {name: report[name] for name in class_names()},
            "confusion_matrix": confusion_matrix(result["labels"], result["predictions"], labels=list(range(4))).tolist(),
        }
        for name in class_names():
            per_class.append({"background": background_name, "class": name, **report[name]})
        rows.append(
            {
                "background": background_name,
                "accuracy": accuracy_score(result["labels"], result["predictions"]),
                "macro_f1": f1_score(
                    result["labels"],
                    result["predictions"],
                    labels=list(range(4)),
                    average="macro",
                    zero_division=0,
                ),
                "prediction_change_rate": np.mean(
                    result["predictions"] != original_predictions
                ),
                "mean_confidence": np.mean(confidence),
                "majority_baseline": np.bincount(result["labels"]).max()
                / len(result["labels"]),
            }
        )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    pd.DataFrame(per_class).to_csv(args.output.with_suffix(".per_class.csv"), index=False)
    val_df.to_csv(args.output.with_suffix(".manifest.csv"), index=False)
    np.savez_compressed(args.output.with_suffix(".predictions.npz"), **{
        f"{condition}_{key}": value for condition, result in results.items() for key, value in result.items()})
    details["metadata"] = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                           "split": args.split, "seed": config.seed, "sample_count": len(val_df),
                           "class_order": class_names(), "conditions": list(conditions),
                           "limitations": "Controlled background stress tests, not field accuracy; leaf masks can have artifacts."}
    args.output.with_suffix(".details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(output.round(4).to_string(index=False))
    print(f"Audit saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
