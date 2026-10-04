from pathlib import Path

import pandas as pd


manifest_path = Path("lesion_manifests/train.csv")
image_dir = Path("data/lesion_hard_negatives/images")
mask_dir = Path("data/lesion_hard_negatives/masks")

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

manifest = pd.read_csv(manifest_path)
new_rows = []

for image_path in sorted(image_dir.iterdir()):
    if image_path.suffix.lower() not in extensions:
        continue

    mask_path = mask_dir / f"{image_path.stem}.png"

    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    new_rows.append(
        {
            "image_path": image_path.as_posix(),
            "mask_path": mask_path.as_posix(),
            "disease": "healthy",
            "source": "hard_negative",
        }
    )

new_records = pd.DataFrame(new_rows)

# Prevent duplicate image entries when the script is run more than once.
existing_paths = set(manifest["image_path"].astype(str))
new_records = new_records[
    ~new_records["image_path"].isin(existing_paths)
]

updated = pd.concat([manifest, new_records], ignore_index=True)
updated.to_csv(manifest_path, index=False)

print(f"Existing rows: {len(manifest)}")
print(f"New rows added: {len(new_records)}")
print(f"Updated rows: {len(updated)}")
