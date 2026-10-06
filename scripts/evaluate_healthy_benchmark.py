"""Evaluate lesion checkpoints on an all-negative healthy benchmark."""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from appleleaf.engine import choose_device
from appleleaf_lesion.dataset import LesionDataset, validate_manifest
from appleleaf_lesion.workflow import load_model


def component_sizes(mask):
    """Return 8-connected component areas for a binary mask."""
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    sizes = []
    height, width = mask.shape
    for row, column in zip(*np.nonzero(mask & ~visited)):
        if visited[row, column]:
            continue
        queue = deque([(row, column)])
        visited[row, column] = True
        size = 0
        while queue:
            current_row, current_column = queue.popleft()
            size += 1
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    next_row = current_row + row_offset
                    next_column = current_column + column_offset
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and mask[next_row, next_column]
                        and not visited[next_row, next_column]
                    ):
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
        sizes.append(size)
    return sizes


def summarize(frame):
    total_pixels = int(frame["pixel_count"].sum())
    total_predicted = int(frame["predicted_pixels"].sum())
    component_sizes_all = [
        size
        for encoded in frame["component_sizes"]
        for size in json.loads(encoded)
    ]
    return {
        "image_count": len(frame),
        "images_with_any_predicted_lesion": int(frame["has_prediction"].sum()),
        "percent_images_with_any_predicted_lesion": 100 * frame["has_prediction"].mean(),
        "false_positive_pixel_rate": total_predicted / total_pixels,
        "mean_predicted_lesion_area_pixels": frame["predicted_pixels"].mean(),
        "mean_predicted_lesion_area_fraction": frame["predicted_fraction"].mean(),
        "total_false_positive_components": int(frame["component_count"].sum()),
        "mean_false_positive_components_per_image": frame["component_count"].mean(),
        "mean_false_positive_component_size_pixels": (
            float(np.mean(component_sizes_all)) if component_sizes_all else 0.0
        ),
        "median_false_positive_component_size_pixels": (
            float(np.median(component_sizes_all)) if component_sizes_all else 0.0
        ),
        "maximum_false_positive_component_size_pixels": (
            int(max(component_sizes_all)) if component_sizes_all else 0
        ),
        "mean_lesion_probability": frame["mean_probability"].mean(),
        "mean_of_image_maximum_lesion_probability": frame["maximum_probability"].mean(),
        "maximum_lesion_probability": frame["maximum_probability"].max(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if len(args.checkpoint) != len(args.name):
        raise SystemExit("Supply one --name for every --checkpoint")
    validate_manifest(args.manifest)
    records = pd.read_csv(args.manifest)
    if not records["empty_mask_verified"].all() or records["source"].ne("evaluation_only").any():
        raise SystemExit("Manifest is not a verified evaluation-only empty-mask benchmark")
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    category_rows = []

    for name, checkpoint_path in zip(args.name, args.checkpoint):
        model, checkpoint = load_model(checkpoint_path, device)
        threshold = float(checkpoint.get("threshold", 0.5))
        dataset = LesionDataset(args.manifest, int(checkpoint["image_size"]))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        rows = []
        model.eval()
        with torch.no_grad():
            for images, masks, paths in loader:
                if masks.count_nonzero():
                    raise SystemExit("Benchmark contains a nonempty lesion mask")
                probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()[:, 0]
                for probability, path in zip(probabilities, paths):
                    predicted = probability >= threshold
                    sizes = component_sizes(predicted)
                    metadata = records.loc[records["image_path"] == path].iloc[0]
                    rows.append(
                        {
                            "checkpoint": name,
                            "image_path": path,
                            "categories": metadata["categories"],
                            "threshold": threshold,
                            "pixel_count": predicted.size,
                            "predicted_pixels": int(predicted.sum()),
                            "predicted_fraction": float(predicted.mean()),
                            "has_prediction": bool(predicted.any()),
                            "component_count": len(sizes),
                            "component_sizes": json.dumps(sizes),
                            "mean_probability": float(probability.mean()),
                            "maximum_probability": float(probability.max()),
                        }
                    )
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output_dir / f"{name}_per_image.csv", index=False)
        summary = {"checkpoint": name, "threshold": threshold, **summarize(frame)}
        summaries.append(summary)
        categories = sorted({item for value in frame["categories"] for item in value.split(";")})
        for category in categories:
            subset = frame[frame["categories"].str.split(";").apply(lambda xs: category in xs)]
            category_rows.append({"checkpoint": name, "category": category, **summarize(subset)})

    summary_frame = pd.DataFrame(summaries)
    category_frame = pd.DataFrame(category_rows)
    summary_frame.to_csv(args.output_dir / "checkpoint_summary.csv", index=False)
    category_frame.to_csv(args.output_dir / "category_summary.csv", index=False)
    (args.output_dir / "results.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
