
from pathlib import Path
from PIL import Image, ImageOps
import pandas as pd
import hashlib
import csv

ROOT = Path("data/plant_pathology_2021")
IMAGES = ROOT / "train_images"
OUTPUT = Path("data/field_manifests/pp2021_audit")
OUTPUT.mkdir(parents=True, exist_ok=True)

records = pd.read_csv(ROOT / "train.csv")
candidates = pd.read_csv(
    ROOT / "pp2021_single_label_candidates.csv"
)

assert len(records) == 18632
assert len(candidates) == 14491
assert records["image"].is_unique

mapping = dict(zip(
    candidates["image"],
    candidates["proposed_label"]
))

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def pixel_sha256(image):
    width, height = image.size
    digest = hashlib.sha256()
    digest.update(f"{width}x{height}|".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()

def dhash256(image):
    gray = image.convert("L").resize(
        (17, 16), Image.Resampling.LANCZOS
    )
    pixels = list(gray.getdata())
    bits = 0
    for y in range(16):
        for x in range(16):
            bits = (bits << 1) | int(
                pixels[y * 17 + x + 1] >
                pixels[y * 17 + x]
            )
    return f"{bits:064x}"

output_file = OUTPUT / "image_inventory.csv"

if output_file.exists():
    raise FileExistsError(
        "Inventory already exists. Preserve it or version a new run."
    )

columns = [
    "image", "original_labels", "proposed_label",
    "status", "width", "height",
    "file_sha256", "pixel_sha256",
    "dhash256", "error"
]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for i, row in enumerate(records.itertuples(index=False), 1):
        filename = str(row.image)
        path = IMAGES / filename

        result = {
            "image": filename,
            "original_labels": row.labels,
            "proposed_label": mapping.get(filename, ""),
            "status": "ok" if filename in mapping else "excluded_label",
            "width": "",
            "height": "",
            "file_sha256": "",
            "pixel_sha256": "",
            "dhash256": "",
            "error": "",
        }

        try:
            if Path(filename).name != filename:
                raise ValueError("Invalid image filename")

            result["file_sha256"] = file_sha256(path)

            with Image.open(path) as raw:
                rgb = ImageOps.exif_transpose(raw).convert("RGB")
                rgb.load()
                result["width"], result["height"] = rgb.size
                result["pixel_sha256"] = pixel_sha256(rgb)
                result["dhash256"] = dhash256(rgb)

        except Exception as exc:
            result["status"] = "image_error"
            result["error"] = str(exc)

        writer.writerow(result)

        if i % 500 == 0:
            print(f"Audited {i}/{len(records)} images", flush=True)

audit = pd.read_csv(output_file)

print("\nFINAL SUMMARY")
print("Total records:", len(audit))
print("Image errors:", int((audit.status == "image_error").sum()))
print("Candidate images:", int(audit.proposed_label.notna().sum()))
print("Unique pixel hashes:", audit.pixel_sha256.nunique())
print("Repeated pixel-hash groups:",
      int((audit.pixel_sha256.value_counts() > 1).sum()))
print("Saved:", output_file)
