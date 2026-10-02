import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from appleleaf.config import ExperimentConfig
from appleleaf.data import attach_segmented_paths, download_dataset, prepare_splits
from appleleaf.datasets import AUDIT_CONDITIONS, BackgroundTestDataset, build_transform
from appleleaf.engine import choose_device, predict
from appleleaf.workflow import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Audit sensitivity to image backgrounds")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/background_audit.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    config = ExperimentConfig(
        image_size=int(checkpoint["image_size"]), seed=int(checkpoint["seed"])
    )
    dataset_path = args.dataset_path or download_dataset()
    _, val_df, _ = prepare_splits(dataset_path, config.seed)
    val_df = attach_segmented_paths(val_df, dataset_path)
    transform = build_transform(config.image_size)
    results = {}
    for background_name in AUDIT_CONDITIONS:
        dataset = BackgroundTestDataset(val_df, background_name, transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        results[background_name] = predict(model, loader, device)

    original_predictions = results["original"]["predictions"]
    rows = []
    for background_name, result in results.items():
        confidence = result["probabilities"].max(axis=1)
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
    print(output.round(4).to_string(index=False))
    print(f"Audit saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
