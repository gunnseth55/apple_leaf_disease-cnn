"""Build a descriptive class breakdown and contact sheets from saved predictions.

No model loading, training, inference, relabeling or threshold tuning.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "artifacts/kashmir_evaluation/efficientnet_b0"
    output = ROOT / "artifacts/kashmir_evaluation/efficientnet_b0_review"
    tracked = ROOT / "results/2026-10-10_efficientnet_b0/kashmir_review"
    for directory in (output, tracked):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "artifacts/efficientnet_b0/apple_efficientnet_b0_checkpoint.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metrics = json.loads((source / "metrics.json").read_text())
    assert digest == metrics["metadata"]["checkpoint_sha256"]
    frame = pd.read_csv(source / "predictions.csv")
    audit = pd.read_csv(ROOT / "results/2026-10-08_kashmir_label_audit/review_manifest.csv")
    frame = frame.merge(audit[["relative_path", "review_id", "draft_status", "conflict_group"]],
                        on="relative_path", how="left", validate="one_to_one")
    assert frame.review_id.notna().all()
    names = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"]
    short = {name: label for name, label in zip(names, ("Scab", "Rot (provisional)", "Cedar rust", "Healthy"))}
    predicted_names = {**short, names[1]: "Black rot"}
    groups = [
        (names[0], names[3], ("highest", "median", "lowest")),
        (names[1], names[3], ("highest", "median")),
        (names[1], names[0], ("highest", "median")),
        (names[3], names[0], ("highest", "median")),
        (names[0], names[1], ("highest",)),
        (names[0], names[2], ("highest",)),
        (names[1], names[1], ("highest",)),  # agreement context, not a verified diagnosis
    ]
    selected = []
    for truth, prediction, ranks in groups:
        candidates = frame[(frame.true_class == truth) & (frame.predicted_class == prediction)]
        candidates = candidates.sort_values(["confidence", "relative_path"], ascending=[False, True])
        for rank in ranks:
            index = {"highest": 0, "median": len(candidates) // 2, "lowest": len(candidates) - 1}[rank]
            row = candidates.iloc[index].to_dict()
            row.update(selection=f"{rank} confidence in folder/prediction group",
                       example_id=f"E{len(selected) + 1:02d}")
            selected.append(row)
    examples = pd.DataFrame(selected)
    assert not examples.relative_path.duplicated().any()
    breakdown = []
    for truth in (names[0], names[1], names[3]):
        group = frame[frame.true_class == truth]
        for prediction in names:
            count = int((group.predicted_class == prediction).sum())
            breakdown.append({"source_class": short[truth], "predicted_class": predicted_names[prediction],
                              "count": count, "source_support": len(group),
                              "percent_of_source_class": 100 * count / len(group)})
    summary = {
        "status": "preliminary; folder diagnoses and rot-to-black-rot mapping unverified",
        "checkpoint_sha256": digest, "checkpoint_unchanged": True,
        "sample_count": len(frame), "review_example_count": len(examples),
        "selection": "Descriptive confidence-stratified examples, not a random statistical sample",
        "scab_recall": 32 / 117, "healthy_recall": 34 / 43,
        "cedar_rust_prediction_count": int((frame.predicted_class == names[2]).sum()),
        "cedar_rust_prediction_rate": float((frame.predicted_class == names[2]).mean()),
        "known_visual_conflict_rows_still_in_original_subset": int(frame.conflict_group.notna().sum()),
        "protocol": "Saved predictions only; no training, inference, diagnoses or exclusions changed",
    }
    # Actual observations from this session's manual contact-sheet inspection.
    # These are not generated diagnoses and must not replace source labels.
    observations = {
        "E01": "Dark upper blade with multiple reddish-brown patches, including a large lower-left patch; pink background.",
        "E02": "Pale underside with prominent veins and two small dark marks; no large brown area apparent at sheet scale.",
        "E03": "Soft-focus blade with distinct tan/brown patches, including one large patch to the right of the midrib; blue background.",
        "E04": "Dark glossy, wrinkled blade on pink background; no conspicuous discrete lesion established at sheet scale.",
        "E05": "Dark blade on pink background, with subtle discoloration and a small light mark; photo alone does not confirm health or black rot.",
        "E06": "Green/yellow blade with large irregular missing/torn region near the upper tip and brown tissue at the upper edge; pink background.",
        "E07": "Blade with holes, a torn upper region and dark reddish-brown margins; known rot/scab conflict group G09.",
        "E08": "Green/pale blade with prominent veins, photographed against textured fabric; no obvious large lesions at sheet scale.",
        "E09": "Pale underside with prominent veins against textured fabric; no obvious large lesions at sheet scale.",
        "E10": "Pale underside with numerous small pale/yellow marks across the blade; appearance does not establish scab or black rot.",
        "E11": "Soft-focus leaf with a broad brown patch on the right side; cedar-rust prediction has only 0.3332 maximum probability.",
        "E12": "Dark blade with extensive brown tissue and missing/torn tissue along the right edge; agreement with rot mapping is not diagnostic confirmation.",
    }
    examples["visual_observation"] = examples.example_id.map(observations)
    examples["reviewer"] = "Codex manual visual review, 2026-10-10"
    examples["diagnosis_status"] = "Unverified; descriptive observation only; no relabeling"
    for directory in (output, tracked):
        examples.to_csv(directory / "representative_examples.csv", index=False)
        pd.DataFrame(breakdown).to_csv(directory / "class_prediction_breakdown.csv", index=False)
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # Two sheets keep individual leaves legible during visual inspection.
    for page, start in enumerate(range(0, len(examples), 6), 1):
        sheet = Image.new("RGB", (1500, 1020), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for slot, (_, row) in enumerate(examples.iloc[start:start + 6].iterrows()):
            x, y = (slot % 3) * 500, (slot // 3) * 510
            with Image.open(row.path) as source_image:
                photo = ImageOps.contain(source_image.convert("RGB"), (480, 385))
            sheet.paste(photo, (x + (500 - photo.width) // 2, y + 100))
            caption = (f"{row.example_id} / {row.review_id}: {Path(row.relative_path).name}\n"
                       f"Folder: {short[row.true_class]} -> Pred: {predicted_names[row.predicted_class]}\n"
                       f"Confidence: {row.confidence:.4f}; {row.selection}\n"
                       f"Audit conflict: {row.conflict_group if pd.notna(row.conflict_group) else 'none recorded'}")
            draw.multiline_text((x + 8, y + 8), caption, fill="black", spacing=4)
        sheet.save(output / f"representative_sheet_{page:02d}.jpg", quality=94)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    print(json.dumps(summary, indent=2))
    print(examples[["example_id", "review_id", "relative_path", "confidence", "conflict_group"]].to_string(index=False))


if __name__ == "__main__":
    main()
