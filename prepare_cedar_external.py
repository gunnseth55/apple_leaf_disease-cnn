from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(
    r"C:\Users\gunn\Desktop\work\appleleaf\data\plant_pathology_2020"
)

IMAGES_DIR = ROOT / "extracted" / "images"

OUTPUT_DIR = Path(
    r"C:\Users\gunn\Desktop\work\appleleaf\data\external_classification\cedar_pp2020"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Find train.csv
# --------------------------------------------------

csv_candidates = list(ROOT.rglob("train.csv"))

if not csv_candidates:
    raise FileNotFoundError(
        f"Could not find train.csv anywhere under:\n{ROOT}"
    )

TRAIN_CSV = csv_candidates[0]

print("Using labels:", TRAIN_CSV)
print("Using images:", IMAGES_DIR)
print("Output:", OUTPUT_DIR)


# --------------------------------------------------
# Read labels
# --------------------------------------------------

df = pd.read_csv(TRAIN_CSV)

print("\nColumns:")
print(df.columns.tolist())


# Plant Pathology 2020 usually contains:
# image_id, healthy, multiple_diseases, rust, scab

required = {"image_id", "rust"}

if not required.issubset(df.columns):
    raise ValueError(
        f"Expected columns {required}, but got:\n{df.columns.tolist()}"
    )


# Keep only clean rust-labelled images.
# This avoids multiple-disease samples.
rust_df = df[df["rust"] == 1].copy()

if "multiple_diseases" in df.columns:
    rust_df = rust_df[rust_df["multiple_diseases"] == 0]


print("\nRust images found:", len(rust_df))


# --------------------------------------------------
# Locate and copy images
# --------------------------------------------------

copied = 0
missing = []

extensions = [".jpg", ".jpeg", ".png"]

for image_id in rust_df["image_id"]:

    source = None

    for ext in extensions:
        candidate = IMAGES_DIR / f"{image_id}{ext}"

        if candidate.exists():
            source = candidate
            break

    if source is None:
        missing.append(image_id)
        continue

    destination = OUTPUT_DIR / source.name

    shutil.copy2(source, destination)
    copied += 1


print("\nFinished.")
print("Copied:", copied)
print("Missing:", len(missing))

if missing:
    print("\nFirst missing IDs:")
    for x in missing[:20]:
        print(x)