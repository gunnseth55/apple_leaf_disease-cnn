"""Audit local PP2020 sources; write metadata only and assign no dataset splits."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

FLAGS = ["healthy", "multiple_diseases", "rust", "scab"]
LABELS = {"scab": ("Apple___Apple_scab", 0), "rust": ("Apple___Cedar_apple_rust", 2),
          "healthy": ("Apple___healthy", 3)}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ID_PATTERN = re.compile(r"Train_\d+\Z")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(path):
    """Record source metadata; normalized images exist only in memory for hashes."""
    row = dict(sha256="", pixel_sha256="", dhash256="", width=None, height=None,
               source_mode="", exif_orientation=None, oriented_width=None, oriented_height=None,
               rgb_compatible=False, image_status="unreadable_image", image_error="")
    try:
        row["sha256"] = sha256(path)  # Preserve a digest even when decoding fails.
        with Image.open(path) as image:
            row.update(width=image.width, height=image.height, source_mode=image.mode)
            image.verify()
        with Image.open(path) as image:
            row.update(width=image.width, height=image.height, source_mode=image.mode,
                       exif_orientation=image.getexif().get(274))
            image.load()  # Strict PIL defaults: do not enable truncated-image loading.
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb.load()
            row.update(oriented_width=rgb.width, oriented_height=rgb.height, rgb_compatible=True)
            row["pixel_sha256"] = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            gray = np.asarray(rgb.convert("L").resize((17, 16), Image.Resampling.LANCZOS))
            row["dhash256"] = np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes().hex()
            row["image_status"] = "ok"
    except Exception as error:
        row["image_error"] = f"{type(error).__name__}: {error}"
    return row


def label_info(record):
    try:
        values = [float(record.get(flag, "")) for flag in FLAGS]
    except (ValueError, TypeError):
        return "invalid_label", None, None
    if any(not np.isfinite(v) or v not in (0, 1) for v in values):
        return "invalid_label", None, None
    if sum(values) != 1:
        return "conflicting_or_empty_label", None, None
    if values[1] == 1:
        return "exclude_multiple_diseases", None, None
    category = next(flag for flag, value in zip(FLAGS, values) if value == 1)
    return "valid_single_label", *LABELS[category]


def read_usage_manifest(project, path, purpose, expected_count):
    if not path.is_file():
        raise FileNotFoundError(f"Required previous-use manifest missing: {path}")
    frame = pd.read_csv(path, dtype=str).fillna("")
    if not {"image_path", "source"}.issubset(frame.columns):
        raise ValueError(f"Previous-use manifest needs image_path and source: {path}")
    source = "hard_negative" if purpose == "segmentation_training" else "evaluation_only"
    frame = frame[frame.source == source]
    if len(frame) != expected_count:
        raise ValueError(f"Expected {expected_count} {source} rows, found {len(frame)}: {path}")
    rows = []
    for value in frame.image_path:
        image = Path(value)
        image = image if image.is_absolute() else project / image
        if not image.is_file():
            raise FileNotFoundError(f"Previous-use image missing: {image}")
        rows.append(dict(image_id=image.stem if ID_PATTERN.fullmatch(image.stem) else "",
                         image_path=str(image.resolve()), purpose=purpose, evidence=str(path.resolve())))
    if len({r["image_path"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate previous-use paths: {path}")
    return rows


def cedar_evidence(project):
    """Collect references for review; a path reference is not proof of evaluation."""
    files = set(project.glob("*.md")) | set(project.glob("*.py"))
    for name in ["artifacts", "results", "scripts"]:
        files.update(p for p in (project / name).rglob("*")
                     if p.suffix.lower() in {".csv", ".json", ".md", ".py", ".ps1"})
    references, errors = [], []
    for path in sorted(files):
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if "cedar_pp2020" in line.lower():
                    references.append(dict(path=str(path.resolve()), line=number, text=line[:600],
                                           evidence_kind="reference_only_not_proof_of_use"))
        except (OSError, UnicodeError) as error:
            errors.append(dict(path=str(path), error=str(error)))
    return references, errors


def audit(project, output, near_distance=4, expected_training=40, expected_eval=50):
    project, output = Path(project).resolve(), Path(output).resolve()
    root = project / "data/plant_pathology_2020"
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the source dataset directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose a new or empty output directory; existing audits are preserved")
    if not 0 <= near_distance <= 256:
        raise ValueError("Near-duplicate distance must be between 0 and 256")
    csvs = list(root.rglob("train.csv"))
    if len(csvs) != 1:
        raise ValueError(f"Expected exactly one labelled train.csv; found {csvs}")
    labels = pd.read_csv(csvs[0], dtype=str).fillna("")
    if not {"image_id", *FLAGS}.issubset(labels.columns):
        raise ValueError(f"Unexpected label columns: {labels.columns.tolist()}")
    images = root / "extracted/images"
    if not images.is_dir():
        raise FileNotFoundError(f"Image directory missing: {images}")
    # Validate the previous-use registry before making any candidate decisions.
    registry = read_usage_manifest(project, project / "lesion_manifests/train.csv", "segmentation_training", expected_training)
    registry += read_usage_manifest(project, project / "data/healthy_structural_benchmark/manifest.csv", "evaluation_only", expected_eval)
    cedar = project / "data/external_classification/cedar_pp2020"
    for path in sorted(cedar.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            registry.append(dict(image_id=path.stem if ID_PATTERN.fullmatch(path.stem) else "",
                                 image_path=str(path.resolve()), purpose="cedar_prior_use_unresolved",
                                 evidence="Extracted copy exists; testing/model-selection history unconfirmed"))
    references, evidence_errors = cedar_evidence(project)
    cache = {}
    def info(path):
        key = str(Path(path).resolve())
        if key not in cache:
            cache[key] = inspect_image(path)
        return cache[key]
    for row in registry:
        row.update(info(row["image_path"]))
        if row["image_status"] != "ok":
            raise ValueError(f"Cannot establish previous-use content identity: {row['image_path']}: {row['image_error']}")
    for purpose in ["segmentation_training", "evaluation_only"]:
        ids = [r["image_id"] for r in registry if r["purpose"] == purpose and r["image_id"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate PP2020 IDs in {purpose} registry")
    lookup = defaultdict(list)
    for path in images.iterdir():
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            lookup[path.stem].append(path)
    id_counts, inventory = Counter(labels.image_id), []
    for index, record in enumerate(labels.to_dict("records")):
        image_id = record["image_id"]
        label_status, label_name, label_index = label_info(record)
        id_status = "ok" if ID_PATTERN.fullmatch(image_id) else "invalid_image_id"
        if id_counts[image_id] > 1:
            id_status = "duplicate_image_id"
        matches = lookup.get(image_id, []) if ID_PATTERN.fullmatch(image_id) else []
        row = dict(source_row=index + 2, image_id=image_id, source="plant_pathology_2020",
                   **{flag: record[flag] for flag in FLAGS}, id_status=id_status,
                   label_status=label_status, label_name=label_name, label=label_index,
                   image_path=str(matches[0].resolve()) if len(matches) == 1 else "",
                   matching_paths=json.dumps([str(p.resolve()) for p in matches]))
        if len(matches) == 1:
            row.update(info(matches[0]))
        else:
            row.update(image_status="missing_image" if not matches else "ambiguous_image",
                       image_error="No unique source image", sha256="", pixel_sha256="", dhash256="")
        inventory.append(row)
        if (index + 1) % 200 == 0:
            print(f"Decoded labelled records: {index + 1}/{len(labels)}", flush=True)
    by_id = defaultdict(list)
    for row in inventory:
        by_id[row["image_id"]].append(row)
    registry_errors = []
    for row in registry:
        sources = by_id.get(row["image_id"], [])
        if row["image_id"] and len(sources) != 1:
            registry_errors.append(f"Unresolved source label identity: {row['image_path']}")
        elif row["image_id"]:
            expected = "Apple___Cedar_apple_rust" if row["purpose"] == "cedar_prior_use_unresolved" else "Apple___healthy"
            if sources[0]["label_name"] != expected:
                registry_errors.append(f"Previous-use label mismatch: {row['image_path']}")
    relations = []
    exact_links, near_links = defaultdict(set), defaultdict(set)
    registry_hashes = [(r, int(r["dhash256"], 16)) for r in registry]
    for index, row in enumerate(inventory):
        # ID-based reservations survive missing/corrupt files as independent provenance.
        exact_links[index].update(r["purpose"] for r in registry
                                  if r["image_id"] and r["image_id"] == row["image_id"])
        if not row.get("dhash256"):
            continue
        current = int(row["dhash256"], 16)
        for previous, other_hash in registry_hashes:
            distance = (current ^ other_hash).bit_count()
            id_match = bool(previous["image_id"]) and previous["image_id"] == row["image_id"]
            exact = row["sha256"] == previous["sha256"] or row["pixel_sha256"] == previous["pixel_sha256"]
            if id_match or exact or distance <= near_distance:
                relations.append(dict(scope="previous_use", left_row=index, left_id=row["image_id"],
                                      right_id=previous["image_id"], right_path=previous["image_path"],
                                      purpose=previous["purpose"], id_match=id_match,
                                      exact_content=exact, dhash_distance=distance))
                (exact_links if id_match or exact else near_links)[index].add(previous["purpose"])
    # Internal duplicate graph preserves groups for future split assignment.
    graph = {i: set() for i, r in enumerate(inventory) if r.get("dhash256")}
    for index in graph:
        row = inventory[index]
        for other_index in graph:
            if other_index >= index:
                break
            other = inventory[other_index]
            distance = (int(row["dhash256"], 16) ^ int(other["dhash256"], 16)).bit_count()
            exact = row["sha256"] == other["sha256"] or row["pixel_sha256"] == other["pixel_sha256"]
            if exact or distance <= near_distance:
                graph[index].add(other_index)
                graph[other_index].add(index)
                relations.append(dict(scope="within_labelled", left_row=index, left_id=row["image_id"],
                                      right_row=other_index, right_id=other["image_id"],
                                      exact_content=exact, dhash_distance=distance))
    visited = set()
    for index in range(len(inventory)):
        if index in visited:
            continue
        group, pending = set(), [index]
        while pending:
            item = pending.pop()
            if item not in group:
                group.add(item)
                pending.extend(graph.get(item, set()) - group)
        visited.update(group)
        uses = set().union(*(exact_links[item] | near_links[item] for item in group))
        # The earlier extraction script selected all clean rust rows. Absence of a
        # particular copied file cannot establish that it was never used.
        if any(inventory[item]["label_name"] == "Apple___Cedar_apple_rust" for item in group):
            uses.add("cedar_prior_use_unresolved")
        group_labels = {inventory[item]["label_name"] for item in group if inventory[item]["label_name"]}
        for item in group:
            row, reasons = inventory[item], []
            row.update(duplicate_group=f"pp2020_{min(group):05d}", duplicate_group_size=len(group),
                       prior_use_roles=";".join(sorted(uses)),
                       previous_segmentation_training="segmentation_training" in uses,
                       reserved_evaluation="evaluation_only" in uses,
                       cedar_prior_use_unresolved="cedar_prior_use_unresolved" in uses,
                       previous_use_registry_complete=not registry_errors and not evidence_errors and cedar.is_dir())
            if row["id_status"] != "ok":
                reasons.append(row["id_status"])
            if row["label_status"] != "valid_single_label":
                reasons.append(row["label_status"])
            if row["image_status"] != "ok":
                reasons.append(row["image_status"])
            if not row["previous_use_registry_complete"]:
                reasons.append("previous_use_registry_incomplete")
            if "evaluation_only" in uses:
                reasons.append("reserved_evaluation_or_duplicate")
            if "cedar_prior_use_unresolved" in uses:
                reasons.append("cedar_previous_use_review_required")
            if len(group) > 1:
                reasons.append("duplicate_review_required")
            if len(group_labels) > 1:
                reasons.append("duplicate_label_conflict")
            row.update(eligible_training_candidate=not reasons,
                       eligible_unseen_evaluation=not reasons and not uses,
                       status="candidate" if not reasons else "quarantined_or_excluded",
                       exclusion_reasons=";".join(reasons))
    unchanged = all(sha256(path) == details["sha256"] for path, details in cache.items() if details["sha256"])
    if not unchanged:
        raise RuntimeError("A source or previous-use image changed during the audit; no candidates written")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(inventory)
    frame.to_csv(output / "pp2020_inventory.csv", index=False)
    frame[frame.eligible_training_candidate].to_csv(output / "training_candidates.csv", index=False)
    frame[~frame.eligible_training_candidate].to_csv(output / "quarantine_and_exclusions.csv", index=False)
    pd.DataFrame(registry).to_csv(output / "previous_use_registry.csv", index=False)
    pd.DataFrame(relations, columns=["scope", "left_row", "left_id", "right_row", "right_id", "right_path", "purpose", "id_match", "exact_content", "dhash_distance"]).to_csv(output / "duplicate_screening.csv", index=False)
    summary = dict(labelled_records=len(frame), training_candidates=int(frame.eligible_training_candidate.sum()),
                   candidate_class_counts=frame[frame.eligible_training_candidate].label_name.value_counts().to_dict(),
                   label_status_counts=frame.label_status.value_counts().to_dict(),
                   image_status_counts=frame.image_status.value_counts().to_dict(),
                   registry_records_by_role=dict(Counter(r["purpose"] for r in registry)),
                   reserved_evaluation_or_duplicate_records=int(frame.reserved_evaluation.sum()),
                   previous_training_or_duplicate_records=int(frame.previous_segmentation_training.sum()),
                   cedar_use_unresolved_or_duplicate_records=int(frame.cedar_prior_use_unresolved.sum()),
                   registry_integrity_errors=registry_errors, evidence_scan_errors=evidence_errors,
                   cedar_use_history_resolved=False, cedar_references=references,
                   duplicate_relations=len(relations), dhash_bits=256, near_duplicate_distance=near_distance,
                   hash_preprocessing="EXIF-oriented RGB pixel SHA; 17x16 grayscale LANCZOS dHash, in memory only",
                   source_labels=str(csvs[0].resolve()), source_labels_sha256=sha256(csvs[0]),
                   challenge_test_used=False, splits_assigned=False, source_images_modified=False,
                   source_and_registry_file_hashes_rechecked=unchanged,
                   notes="Cedar extraction does not prove testing/model selection; affected images are held for review. Prior training images cannot be unseen evaluation. Near-duplicate hits require review; no splits assigned.")
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cedar_references"}, indent=2))
    return frame, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    args = parser.parse_args(argv)
    audit(args.project, args.output_dir or args.project / "data/field_manifests/pp2020_audit", args.near_duplicate_distance)


if __name__ == "__main__":
    main()
