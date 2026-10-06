"""Build and audit the evaluation-only healthy structural-negative benchmark."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# Manually reviewed healthy images. Categories are multi-label and describe
# structures that can plausibly be confused with lesions.
SELECTION = {
    "Train_1001": "stem;vein;natural_background;edge",
    "Train_1012": "shadow;sunlight;vein;natural_background",
    "Train_1034": "fold;sunlight;natural_background;edge",
    "Train_1085": "fold;shadow;vein;natural_background",
    "Train_1092": "hole;stem;sunlight;natural_background",
    "Train_1101": "stem;vein;natural_background;edge",
    "Train_1133": "stem;shadow;natural_background;edge",
    "Train_1160": "hole;edge_damage;sunlight;natural_background",
    "Train_1195": "shadow;vein;natural_background;edge",
    "Train_1224": "fold;sunlight;natural_background;edge",
    "Train_1265": "hole;edge_damage;natural_background;edge",
    "Train_1285": "shadow;sunlight;vein;natural_background",
    "Train_1341": "stem;shadow;sunlight;natural_background",
    "Train_1368": "stem;shadow;natural_background;edge",
    "Train_1388": "fold;edge_damage;shadow;natural_background",
    "Train_1397": "hole;fold;sunlight;natural_background",
    "Train_1415": "fold;vein;sunlight;natural_background",
    "Train_1457": "shadow;sunlight;edge;natural_background",
    "Train_1478": "fold;stem;natural_background;edge",
    "Train_1513": "stem;sunlight;natural_background;edge",
    "Train_1557": "fold;vein;sunlight;natural_background",
    "Train_1608": "shadow;vein;natural_background;edge",
    "Train_1646": "fold;sunlight;natural_background;edge",
    "Train_1681": "fold;stem;sunlight;natural_background",
    "Train_1723": "stem;shadow;natural_background;edge",
    "Train_1750": "shadow;sunlight;vein;natural_background",
    "Train_1803": "stem;shadow;natural_background;edge",
    "Train_249": "fold;edge_damage;sunlight;natural_background",
    "Train_263": "stem;sunlight;natural_background;edge",
    "Train_329": "fold;shadow;natural_background;edge",
    "Train_335": "stem;shadow;natural_background;edge",
    "Train_373": "hole;edge_damage;sunlight;natural_background",
    "Train_404": "fold;stem;natural_background;edge",
    "Train_412": "stem;sunlight;natural_background;edge",
    "Train_421": "stem;vein;shadow;natural_background",
    "Train_46": "stem;sunlight;natural_background;edge",
    "Train_478": "fold;sunlight;natural_background;edge",
    "Train_502": "fold;vein;shadow;natural_background",
    "Train_526": "fold;stem;shadow;natural_background",
    "Train_551": "hole;edge_damage;shadow;natural_background",
    "Train_569": "fold;sunlight;natural_background;edge",
    "Train_621": "fold;shadow;natural_background;edge",
    "Train_626": "fold;vein;sunlight;natural_background",
    "Train_646": "stem;shadow;sunlight;natural_background",
    "Train_657": "shadow;sunlight;natural_background;edge",
    "Train_800": "fold;vein;natural_background;edge",
    "Train_849": "fold;shadow;natural_background;edge",
    "Train_863": "fold;sunlight;natural_background;edge",
    "Train_918": "shadow;sunlight;natural_background;edge",
    "Train_934": "fold;edge_damage;sunlight;natural_background",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(path, size=16):
    with Image.open(path) as source:
        gray = source.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.int16)
    return np.packbits(values[:, 1:] > values[:, :-1]).tobytes()


def hamming(left, right):
    return sum(int(byte).bit_count() for byte in bytes(a ^ b for a, b in zip(left, right)))


def image_paths_from_manifest(path):
    return [Path(value) for value in pd.read_csv(path)["image_path"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels).set_index("image_id")
    sources = {path.stem: path for path in args.candidate_dir.iterdir() if path.is_file()}
    training_paths = image_paths_from_manifest(args.training_manifest)
    selected_paths = [sources[name] for name in SELECTION]
    if len(selected_paths) != len(set(selected_paths)):
        raise SystemExit("Selection contains duplicate paths")

    invalid_labels = [
        name
        for name in SELECTION
        if name not in labels.index or int(labels.loc[name, "healthy"]) != 1
    ]
    if invalid_labels:
        raise SystemExit(f"Selection contains non-healthy or unlabelled rows: {invalid_labels}")

    train_sha = {sha256(path): path for path in training_paths}
    selected_sha = {}
    exact_conflicts = []
    for path in selected_paths:
        digest = sha256(path)
        if digest in train_sha:
            exact_conflicts.append((path, train_sha[digest]))
        if digest in selected_sha:
            exact_conflicts.append((path, selected_sha[digest]))
        selected_sha[digest] = path
    if exact_conflicts:
        raise SystemExit(f"Exact duplicate conflicts: {exact_conflicts}")

    train_hashes = [(path, difference_hash(path)) for path in training_paths]
    selected_hashes = [(path, difference_hash(path)) for path in selected_paths]
    audit_rows = []
    near_conflicts = []
    for index, (path, digest) in enumerate(selected_hashes):
        comparisons = train_hashes + selected_hashes[:index]
        nearest_path, nearest_distance = min(
            ((other, hamming(digest, other_hash)) for other, other_hash in comparisons),
            key=lambda item: item[1],
        )
        audit_rows.append(
            {
                "image_id": path.stem,
                "sha256": sha256(path),
                "nearest_image": str(nearest_path),
                "dhash_distance": nearest_distance,
                "near_duplicate_threshold": args.near_duplicate_distance,
                "passed": nearest_distance > args.near_duplicate_distance,
            }
        )
        if nearest_distance <= args.near_duplicate_distance:
            near_conflicts.append((path, nearest_path, nearest_distance))
    if near_conflicts:
        raise SystemExit(f"Near-duplicate conflicts: {near_conflicts}")

    image_dir = args.output_dir / "images"
    mask_dir = args.output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for path in selected_paths:
        destination = image_dir / path.name
        shutil.copy2(path, destination)
        with Image.open(path) as source:
            width, height = source.size
        mask_path = mask_dir / f"{path.stem}.png"
        Image.new("L", (width, height), 0).save(mask_path)
        if np.asarray(Image.open(mask_path)).any():
            raise SystemExit(f"Nonempty generated mask: {mask_path}")
        manifest_rows.append(
            {
                "image_path": destination.as_posix(),
                "mask_path": mask_path.as_posix(),
                "disease": "healthy",
                "source": "evaluation_only",
                "categories": SELECTION[path.stem],
                "healthy_label_verified": True,
                "empty_mask_verified": True,
            }
        )

    pd.DataFrame(manifest_rows).to_csv(args.output_dir / "manifest.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(args.output_dir / "duplicate_audit.csv", index=False)
    category_counts = {}
    for categories in SELECTION.values():
        for category in categories.split(";"):
            category_counts[category] = category_counts.get(category, 0) + 1
    report = {
        "purpose": "evaluation_only",
        "image_count": len(SELECTION),
        "source_dataset": "Plant Pathology 2020 healthy-labelled rows",
        "healthy_labels_verified": True,
        "empty_masks_verified": True,
        "training_exact_duplicates": 0,
        "training_or_internal_near_duplicates": 0,
        "difference_hash_bits": 256,
        "near_duplicate_hamming_threshold": args.near_duplicate_distance,
        "category_counts": dict(sorted(category_counts.items())),
    }
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
