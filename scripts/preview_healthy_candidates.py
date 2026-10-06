"""Create labelled contact sheets for manual healthy-benchmark curation."""

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-page", type=int, default=80)
    args = parser.parse_args()

    training = pd.read_csv(args.training_manifest)
    used = {Path(value).stem.lower() for value in training["image_path"]}
    candidates = [
        path
        for path in sorted(args.candidate_dir.iterdir())
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and path.stem.lower() not in used
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    columns, tile_width, tile_height = 8, 160, 145
    rows = math.ceil(args.per_page / columns)
    for page_index in range(math.ceil(len(candidates) / args.per_page)):
        page = candidates[
            page_index * args.per_page : (page_index + 1) * args.per_page
        ]
        sheet = Image.new(
            "RGB", (columns * tile_width, rows * tile_height), "black"
        )
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(page):
            with Image.open(path) as source:
                image = ImageOps.fit(source.convert("RGB"), (tile_width, 118))
            x = (index % columns) * tile_width
            y = (index // columns) * tile_height
            sheet.paste(image, (x, y))
            draw.text((x + 3, y + 121), path.stem, fill="white")
        sheet.save(args.output_dir / f"candidates_{page_index + 1:02d}.jpg")
    print(f"Rendered {len(candidates)} unused candidates across contact sheets")


if __name__ == "__main__":
    main()
