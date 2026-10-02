import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from appleleaf.engine import choose_device
from appleleaf_lesion.dataset import LesionDataset, validate_manifest
from appleleaf_lesion.engine import metrics_from_totals, segmentation_totals
from appleleaf_lesion.workflow import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and visualize lesion masks")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lesion_evaluation"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def denormalize(image):
    mean = np.asarray((0.485, 0.456, 0.406))
    standard_deviation = np.asarray((0.229, 0.224, 0.225))
    return np.clip(image.transpose(1, 2, 0) * standard_deviation + mean, 0, 1)


def save_example(image, truth, probability, source, output):
    predicted = probability >= 0.5
    overlay = image.copy()
    overlay[predicted] = 0.55 * overlay[predicted] + 0.45 * np.array([1.0, 0.0, 0.0])
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    panels = (image, truth, probability, overlay)
    titles = ("Image", "Ground-truth lesion", "Lesion probability", "Prediction overlay")
    for axis, panel, title in zip(axes, panels, titles):
        axis.imshow(panel, cmap="magma" if panel.ndim == 2 else None, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(Path(source).name)
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    validate_manifest(args.manifest)
    dataset = LesionDataset(args.manifest, int(checkpoint["image_size"]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    threshold = float(checkpoint.get("threshold", 0.5))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tp = fp = fn = 0
    example_number = 0
    model.eval()
    with torch.no_grad():
        for images, masks, paths in loader:
            logits = model(images.to(device)).cpu()
            batch_tp, batch_fp, batch_fn = segmentation_totals(logits, masks, threshold)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            probabilities = torch.sigmoid(logits).numpy()
            for image, mask, probability, source in zip(
                images.numpy(), masks.numpy(), probabilities, paths
            ):
                if example_number >= args.examples:
                    continue
                save_example(
                    denormalize(image),
                    mask[0],
                    probability[0],
                    source,
                    args.output_dir / f"example_{example_number:03d}.png",
                )
                example_number += 1
    metrics = metrics_from_totals(tp, fp, fn)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(" | ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    print(f"Evaluation saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
