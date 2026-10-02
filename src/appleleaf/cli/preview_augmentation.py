import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from appleleaf.config import ExperimentConfig
from appleleaf.data import attach_segmented_paths, download_dataset, prepare_splits
from appleleaf.datasets import load_image_and_mask, replace_background


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview the background transformations used during training"
    )
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/augmentation_preview.png")
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def smooth_texture(shape, rng):
    small = Image.fromarray(rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8))
    return np.asarray(
        small.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    )


def main():
    args = parse_args()
    dataset_path = args.dataset_path or download_dataset()
    config = ExperimentConfig(seed=args.seed)
    train_df, _, _ = prepare_splits(dataset_path, config.seed)
    train_df = attach_segmented_paths(train_df, dataset_path)
    examples = train_df.groupby("label", sort=True).head(1)
    rng = np.random.default_rng(args.seed)
    column_names = ["Original", "Leaf mask", "White", "Blue", "Black", "Random texture"]
    figure, axes = plt.subplots(len(examples), len(column_names), figsize=(16, 10))

    for row_number, (_, row) in enumerate(examples.iterrows()):
        original, mask = load_image_and_mask(row["path"], row["segmented_path"])
        texture = smooth_texture(original.shape, rng)
        variants = [
            original,
            mask,
            replace_background(original, mask, (255, 255, 255)),
            replace_background(original, mask, (60, 120, 200)),
            replace_background(original, mask, (0, 0, 0)),
            np.where(mask[..., None], original, texture).astype(np.uint8),
        ]
        for column, (axis, variant) in enumerate(zip(axes[row_number], variants)):
            axis.imshow(variant, cmap="gray" if column == 1 else None)
            axis.axis("off")
            if row_number == 0:
                axis.set_title(column_names[column])
            if column == 0:
                axis.text(
                    -0.08,
                    0.5,
                    row["class_name"].replace("Apple___", "").replace("_", " "),
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                )

    figure.suptitle(
        "Training background augmentation examples\n"
        "Leaf pixels stay unchanged; only pixels outside the leaf mask are replaced"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"Augmentation preview saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
