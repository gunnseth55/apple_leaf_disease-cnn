"""Descriptive image audit, not disease diagnosis or model inference."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]


def image_stats(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        grey = np.asarray(image.resize((128, 128)).convert("L"), dtype=np.float32)
    laplacian = (grey[1:-1, :-2] + grey[1:-1, 2:] + grey[:-2, 1:-1] + grey[2:, 1:-1]
                 - 4 * grey[1:-1, 1:-1])
    return {"width": width, "height": height, "aspect_ratio": width / height,
            "luminance_mean": float(grey.mean()), "luminance_std": float(grey.std()),
            "high_frequency_variance_128": float(laplacian.var())}


def main():
    output = ROOT / "artifacts/domain_robustness_analysis"
    tracked = ROOT / "results/2026-10-10_robustness_plan"
    output.mkdir(parents=True, exist_ok=True)
    tracked.mkdir(parents=True, exist_ok=True)
    pv = pd.read_csv(ROOT / "artifacts/efficientnet_b0/train_manifest.csv")
    kashmir = pd.read_csv(ROOT / "artifacts/kashmir_evaluation/efficientnet_b0/manifest.csv")
    records, examples = [], []
    for domain, frame in (("PlantVillage train", pv), ("Kashmir external", kashmir)):
        for _, row in frame.iterrows():
            records.append({"domain": domain, "path": row.path, "label": int(row.label),
                            **image_stats(row.path)})
        # Three per represented class, ordered by filename: early/middle/late.
        # Qualitative selection is independent of model correctness/confidence.
        selected = []
        for label, group in frame.groupby("label", sort=True):
            group = group.sort_values("path")
            for index in (0, len(group) // 2, len(group) - 1):
                row = group.iloc[index]
                selected.append({"domain": domain, "label": int(label), "path": row.path,
                                 "image_id": f"{'PV' if domain.startswith('Plant') else 'K'}{len(selected) + 1:02d}"})
        examples.extend(selected)
        for page, start in enumerate(range(0, len(selected), 6), 1):
            sheet = Image.new("RGB", (1500, 990), "#eeeeee")
            draw = ImageDraw.Draw(sheet)
            for slot, row in enumerate(selected[start:start + 6]):
                x, y = slot % 3 * 500, slot // 3 * 495
                with Image.open(row["path"]) as image:
                    original_size = image.size
                    photo = ImageOps.contain(image.convert("RGB"), (480, 405))
                sheet.paste(photo, (x + (500 - photo.width) // 2, y + 75))
                draw.multiline_text((x + 8, y + 8),
                    f"{row['image_id']}: {domain}; folder class {row['label']}\n"
                    f"{Path(row['path']).name[:66]}\nOriginal pixels: {original_size}", fill="black", spacing=4)
            slug = "plantvillage" if domain.startswith("Plant") else "kashmir"
            sheet.save(output / f"{slug}_{page:02d}.jpg", quality=94)
    frame = pd.DataFrame(records)
    measures = ["width", "height", "aspect_ratio", "luminance_mean", "luminance_std", "high_frequency_variance_128"]
    summary = {}
    for domain, group in frame.groupby("domain"):
        summary[domain] = {"count": len(group), "measurements": {
            name: {"median": float(group[name].median()),
                   "p10": float(group[name].quantile(.1)), "p90": float(group[name].quantile(.9))}
            for name in measures}}
    summary["limitations"] = [
        "Luminance includes background and does not isolate illumination of the leaf.",
        "High-frequency variance is confounded by veins, lesions, texture and compression; not a standalone blur measurement.",
        "No acquisition IDs, calibrated disease severity or expert-confirmed Kashmir diagnoses.",
        "Kashmir was previously inspected; it is evaluation-only, not a blind unseen benchmark.",
    ]
    frame.to_csv(tracked / "image_measurements.csv", index=False)
    pd.DataFrame(examples).to_csv(tracked / "visual_sample_manifest.csv", index=False)
    (tracked / "domain_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
