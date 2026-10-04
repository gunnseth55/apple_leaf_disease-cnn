
from pathlib import Path
import shutil

import pandas as pd


root = Path("data/plant_pathology_2020")
csv_path = root / "plant_pathology_labels.csv"
extracted = root / "extracted"
destination = root / "healthy_candidates"

destination.mkdir(parents=True, exist_ok=True)

labels = pd.read_csv(csv_path)
print("CSV columns:", list(labels.columns))

healthy = labels[labels["healthy"] == 1]
image_column = "Image" if "Image" in labels.columns else "image_id"

available = {
    path.stem.lower(): path
    for path in extracted.rglob("*")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
}

copied = 0
missing = []

for image_id in healthy[image_column].astype(str):
    source = available.get(Path(image_id).stem.lower())

    if source is None:
        missing.append(image_id)
        continue

    shutil.copy2(source, destination / source.name)
    copied += 1

print(f"Healthy rows: {len(healthy)}")
print(f"Images copied: {copied}")
print(f"Missing images: {len(missing)}")
print("Missing examples:", missing[:10])

