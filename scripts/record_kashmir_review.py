"""Record this session's visual decisions and assemble an unverified draft manifest."""
import csv
import json
from collections import Counter
from pathlib import Path

out = Path("artifacts/kashmir_label_audit/visual")
dest = Path("results/2026-10-08_kashmir_label_audit")
dest.mkdir(exist_ok=False)
with (out / "manifest.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
with (out / "cross_folder_candidates.csv").open(newline="", encoding="utf-8") as stream:
    pairs = list(csv.DictReader(stream))
accepted = {"P01", "P02", "P03", "P04", "P05", "P06", "P07", "P09", "P10", "P11", "P12", "P84"}
for pair in pairs:
    pair["status"] = "visually_same_or_nearly_identical_photo_conflicting_folder_labels" if pair["pair_id"] in accepted else "visual_duplicate_not_established"
    pair["reviewer"] = "Codex visual review; not a plant pathology diagnosis"
    pair["review_date"] = "2026-10-08"
with (dest / "cross_folder_pair_review.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
groups = [
    ["K001", "K129"], ["K002", "K145"], ["K003", "K151"],
    ["K017", "K176"], ["K046", "K228", "K313"], ["K047", "K229"],
    ["K049", "K230"], ["K050", "K231"], ["K051", "K232"],
    ["K052", "K233"], ["K072", "K075", "K278", "K279"],
]
conflict_ids = {rid for group in groups for rid in group}
keep_hashes = set()
for row in rows:
    row["visual_screened"] = True
    row["label_status"] = "folder_derived_unverified"
    row["confirmed_diagnosis"] = ""
    row["reviewer_note"] = "Thumbnail screening; no pathogen diagnosis or absence-of-disease confirmation."
    if row["review_id"] in conflict_ids:
        row["draft_status"] = "exclude_conflicting_folder_labels"
    elif row["pixel_sha256"] in keep_hashes:
        row["draft_status"] = "exclude_exact_same_class_duplicate"
    else:
        row["draft_status"] = "candidate_pending_label_verification"
        keep_hashes.add(row["pixel_sha256"])
    row["conflict_group"] = next((f"G{i+1:02}" for i,g in enumerate(groups) if row["review_id"] in g), "")
with (dest / "review_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
summary = {"reviewed_images": len(rows), "reviewed_class_sheets": 14,
           "cross_folder_candidates_reviewed": len(pairs), "accepted_visual_conflict_pairs": len(accepted),
           "exact_cross_class_conflict_groups": 3, "additional_visual_conflict_groups": 8,
           "total_conflicting_rows": len(conflict_ids),
           "exact_same_class_duplicate_extra_rows": 5,
           "draft_candidates": sum(r["draft_status"].startswith("candidate") for r in rows),
           "draft_candidate_folder_counts": dict(Counter(r["folder_label"] for r in rows if r["draft_status"].startswith("candidate"))),
           "rot_mapping_confirmed": False,
           "pathologist_review_performed": False,
           "benchmark_ready": False,
           "notes": "Visual screening and conservative conflict exclusions do not validate disease identity. Near-duplicate search is not exhaustive; no source-leaf IDs are available."}
(dest / "review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
for source,target in [(out/"audit.json",dest/"image_audit.json"),
                       (Path("artifacts/kashmir_label_audit/recovery/recovery.json"),dest/"recovery.json")]:
    target.write_bytes(source.read_bytes())
print(json.dumps(summary, indent=2))
