import argparse
import os
from pathlib import Path

import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create manifests for the Zenodo Apple lesion dataset"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        help="optional Synthetic_Dataset root; added to training only",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("lesion_manifests"))
    return parser.parse_args()


def files_by_stem(folder):
    result = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in result:
                raise RuntimeError(f"Duplicate basename in {folder}: {path.stem}")
            result[path.stem] = path.resolve()
    return result


def disease_from_stem(stem):
    return stem.rsplit("_", 1)[0]


def portable_relative_path(path):
    return Path(os.path.relpath(path, Path.cwd())).as_posix()


def main():
    args = parse_args()
    root = args.dataset_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        images = files_by_stem(root / "images" / split)
        masks = files_by_stem(root / "masks" / split)
        if images.keys() != masks.keys():
            raise RuntimeError(
                f"Unmatched {split} files; image-only={sorted(images.keys() - masks.keys())[:5]}, "
                f"mask-only={sorted(masks.keys() - images.keys())[:5]}"
            )
        rows = []
        for stem in sorted(images):
            with Image.open(images[stem]) as image, Image.open(masks[stem]) as mask:
                if image.size != mask.size:
                    raise RuntimeError(f"Size mismatch for {stem}: {image.size} vs {mask.size}")
            rows.append(
                {
                    "image_path": portable_relative_path(images[stem]),
                    "mask_path": portable_relative_path(masks[stem]),
                    "disease": disease_from_stem(stem),
                    "source": "real",
                }
            )
        if split == "train" and args.synthetic_root:
            synthetic_root = args.synthetic_root.resolve()
            for disease_folder in sorted((synthetic_root / "stable_diffusion_images").iterdir()):
                if not disease_folder.is_dir():
                    continue
                synthetic_images = files_by_stem(disease_folder)
                synthetic_masks = files_by_stem(
                    synthetic_root / "masks" / disease_folder.name
                )
                if synthetic_images.keys() != synthetic_masks.keys():
                    raise RuntimeError(f"Unmatched synthetic files for {disease_folder.name}")
                for stem in sorted(synthetic_images):
                    with Image.open(synthetic_images[stem]) as image, Image.open(
                        synthetic_masks[stem]
                    ) as mask:
                        if image.size != mask.size:
                            raise RuntimeError(f"Synthetic size mismatch for {stem}")
                    rows.append(
                        {
                            "image_path": portable_relative_path(
                                synthetic_images[stem]
                            ),
                            "mask_path": portable_relative_path(
                                synthetic_masks[stem]
                            ),
                            "disease": disease_folder.name,
                            "source": "synthetic",
                        }
                    )
        output = args.output_dir / f"{split}.csv"
        pd.DataFrame(rows).to_csv(output, index=False)
        counts = pd.Series(row["disease"] for row in rows).value_counts().to_dict()
        print(f"{split}: {len(rows)} pairs {counts} -> {output.resolve()}")


if __name__ == "__main__":
    main()
