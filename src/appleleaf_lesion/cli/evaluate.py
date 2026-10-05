import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from appleleaf.engine import choose_device
from appleleaf_lesion.dataset import LesionDataset, validate_manifest
from appleleaf_lesion.engine import (
    metrics_from_totals,
    metrics_per_sample,
    segmentation_totals,
)
from appleleaf_lesion.workflow import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and visualize lesion masks")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lesion_evaluation"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the probability threshold stored in the checkpoint",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def denormalize(image):
    mean = np.asarray((0.485, 0.456, 0.406))
    standard_deviation = np.asarray((0.229, 0.224, 0.225))
    return np.clip(image.transpose(1, 2, 0) * standard_deviation + mean, 0, 1)


def save_example(image, truth, probability, source, output, threshold=0.5):
    predicted = probability >= threshold
    overlay = image.copy()
    overlay[predicted] = 0.55 * overlay[predicted] + 0.45 * np.array([1.0, 0.0, 0.0])
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    panels = (image, truth, probability, overlay)
    titles = (
        "Image",
        "Ground-truth lesion",
        "Lesion probability",
        f"Prediction overlay (p >= {threshold:.2f})",
    )
    for axis, panel, title in zip(axes, panels, titles):
        axis.imshow(panel, cmap="magma" if panel.ndim == 2 else None, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(Path(source).name)
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def summarize_by_disease(per_image):
    rows = []
    for disease, group in per_image.groupby("disease", sort=True):
        totals = group[["true_positive", "false_positive", "false_negative"]].sum()
        pooled = metrics_from_totals(
            totals["true_positive"],
            totals["false_positive"],
            totals["false_negative"],
        )
        rows.append(
            {
                "disease": disease,
                "image_count": len(group),
                **pooled,
                "mean_image_dice": group["dice"].mean(),
                "median_image_dice": group["dice"].median(),
                "std_image_dice": group["dice"].std(ddof=0),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    validate_manifest(args.manifest)
    dataset = LesionDataset(args.manifest, int(checkpoint["image_size"]))
    if "disease" not in dataset.records.columns:
        raise SystemExit(
            "Per-disease evaluation requires a 'disease' column in the manifest"
        )
    disease_by_path = dict(
        zip(
            dataset.records["image_path"].astype(str),
            dataset.records["disease"].astype(str),
        )
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint.get("threshold", 0.5))
    )
    if not 0 <= threshold <= 1:
        raise SystemExit("Threshold must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tp = fp = fn = 0
    per_image_rows = []
    example_number = 0
    model.eval()
    with torch.no_grad():
        for images, masks, paths in loader:
            logits = model(images.to(device)).cpu()
            batch_tp, batch_fp, batch_fn = segmentation_totals(logits, masks, threshold)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            sample_metrics = metrics_per_sample(logits, masks, threshold)
            probabilities = torch.sigmoid(logits).numpy()
            for image, target, probability, source, sample_metric, logit in zip(
                images.numpy(),
                masks,
                probabilities,
                paths,
                sample_metrics,
                logits,
            ):
                sample_tp, sample_fp, sample_fn = segmentation_totals(
                    logit, target, threshold
                )
                per_image_rows.append(
                    {
                        "image_path": source,
                        "disease": disease_by_path[source],
                        **sample_metric,
                        "true_positive": sample_tp,
                        "false_positive": sample_fp,
                        "false_negative": sample_fn,
                    }
                )
                if example_number >= args.examples:
                    continue
                save_example(
                    denormalize(image),
                    target.numpy()[0],
                    probability[0],
                    source,
                    args.output_dir / f"example_{example_number:03d}.png",
                    threshold,
                )
                example_number += 1
    per_image = pd.DataFrame(per_image_rows)
    per_disease = summarize_by_disease(per_image)
    per_image.to_csv(args.output_dir / "per_image_metrics.csv", index=False)
    per_disease.to_csv(args.output_dir / "per_disease_metrics.csv", index=False)

    metrics = metrics_from_totals(tp, fp, fn)
    metrics.update(
        {
            "mean_per_image_dice": per_image["dice"].mean(),
            "median_per_image_dice": per_image["dice"].median(),
            "std_per_image_dice": per_image["dice"].std(ddof=0),
        }
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(" | ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    print("Per-disease results:")
    for row in per_disease.itertuples(index=False):
        print(
            f"  {row.disease}: dice={row.dice:.4f} | iou={row.iou:.4f} | "
            f"precision={row.precision:.4f} | recall={row.recall:.4f} | "
            f"mean-image-dice={row.mean_image_dice:.4f}"
        )
    print("Worst five images by Dice:")
    for row in per_image.nsmallest(5, "dice").itertuples(index=False):
        print(f"  {row.dice:.4f} | {row.disease} | {row.image_path}")
    print(f"Evaluation saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
