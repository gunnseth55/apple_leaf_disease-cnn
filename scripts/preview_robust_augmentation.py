"""Preview training augmentations without loading a model or running epochs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps

from appleleaf.data import attach_segmented_paths
from appleleaf.robustness import RobustBackgroundDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=Path("artifacts/efficientnet_b0/train_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/domain_robustness_analysis/augmentation_preview.jpg"))
    args = parser.parse_args()
    torch.manual_seed(42)
    frame = attach_segmented_paths(pd.read_csv(args.train_manifest), args.dataset_path)
    dataset = RobustBackgroundDataset(frame)
    sheet = Image.new("RGB", (1250, 1080), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    for row_index in range(4):
        index = frame.index[frame.label == row_index][0]
        with Image.open(frame.iloc[index].path) as image:
            original = ImageOps.contain(image.convert("RGB"), (240, 240))
        sheet.paste(original, (0, row_index * 270 + 25))
        draw.text((5, row_index * 270 + 5), f"Original: class {row_index}", fill="black")
        for column in range(1, 5):
            tensor, _ = dataset[index]
            rgb = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            image = Image.fromarray(np.round(rgb * 255).astype(np.uint8)).resize((240, 240))
            sheet.paste(image, (column * 250, row_index * 270 + 25))
            draw.text((column * 250 + 5, row_index * 270 + 5), f"Robust sample {column}", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=94)
    print(f"Saved {args.output}; no model loaded or training performed")


if __name__ == "__main__":
    main()
