"""C2-B: quarantine conflicting PP2021 records using saved audits, without image decoding."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile

import pandas as pd


MAPPING = {"scab": "Apple___Apple_scab", "frog_eye_leaf_spot": "Apple___Black_rot",
           "rust": "Apple___Cedar_apple_rust", "healthy": "Apple___healthy"}
EXPECTED = dict(original_records=18632, original_candidates=14491, unique_pixels=18605,
                conflicting_groups=27, quarantined_records=54, quarantined_candidates=27,
                remaining_candidates=14464)
REASON = "exact_pixel_duplicate_conflicting_original_labels"
QUARANTINE = "quarantined_label_conflict"
PENDING = "pending_cross_dataset_audit"
OUTPUTS = ("quarantined_conflicting_records.csv", "quarantined_conflicting_candidates.csv",
           "candidates_after_internal_audit.csv", "internal_quarantine_report.json")
SOURCES = {"labels": "data/plant_pathology_2021/train.csv",
           "candidates": "data/plant_pathology_2021/pp2021_single_label_candidates.csv",
           "inventory": "data/field_manifests/pp2021_audit/image_inventory.csv",
           "groups": "data/field_manifests/pp2021_audit/internal_duplicate_groups.csv",
           "members": "data/field_manifests/pp2021_audit/internal_duplicate_records.csv"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def flags(series):
    normalized = series.astype(str).str.lower()
    require(normalized.isin(["true", "false"]).all(), "Invalid conflict boolean")
    return normalized.eq("true")


def counts(frame):
    return {str(key): int(value) for key, value in sorted(Counter(frame.labels).items())}


def build_outputs(tables, expected=None):
    """Pure metadata transformation. Small alternative expectations are for unit fixtures only."""
    expected = EXPECTED if expected is None else expected
    labels, candidates, inventory, groups, members = [tables[key].copy() for key in SOURCES]
    schemas = [(labels, {"image", "labels"}), (candidates, {"image", "labels", "proposed_label", "review_status"}),
               (inventory, {"image", "original_labels", "proposed_label", "status", "width", "height",
                            "file_sha256", "pixel_sha256", "dhash256", "error"}),
               (groups, {"group_id", "image_count", "image_ids", "original_labels", "label_conflict",
                         "candidate_count", "excluded_count", "status"}),
               (members, {"image", "group_id", "label_conflict", *inventory.columns})]
    for frame, required in schemas:
        require(required <= set(frame), f"Missing required columns: {sorted(required - set(frame))}")
    require(len(labels) == expected["original_records"], "Unexpected original record count")
    require(len(candidates) == expected["original_candidates"], "Unexpected original candidate count")
    for frame in [labels, candidates, inventory, members]:
        require(frame.image.is_unique and frame.image.ne("").all(), "Duplicate/empty image IDs")
        require(frame.image.map(lambda value: Path(value).name == value and "/" not in value and "\\" not in value).all(),
                "Unsafe image filename")
    require(groups.group_id.is_unique and groups.group_id.ne("").all(), "Duplicate/empty group IDs")
    require(set(inventory.image) == set(labels.image), "Inventory/original ID mismatch")
    original = labels.set_index("image").labels
    require(inventory.original_labels.eq(inventory.image.map(original)).all(), "Inventory original labels changed")
    require(candidates.labels.eq(candidates.image.map(original)).all(), "Candidate original labels changed")
    approved = labels[labels.labels.isin(MAPPING)]
    require(set(approved.image) == set(candidates.image), "Candidate IDs do not match original single-label categories")
    require(candidates.proposed_label.eq(candidates.labels.map(MAPPING)).all(), "Candidate proposed labels changed")
    proposed = candidates.set_index("image").proposed_label
    require(inventory.proposed_label.eq(inventory.image.map(proposed).fillna("")).all(), "Inventory proposed labels changed")
    require(inventory.status.eq(inventory.image.isin(candidates.image).map({True: "ok", False: "excluded_label"})).all()
            and inventory.error.eq("").all(), "Inventory contains image errors or inconsistent statuses")
    for column in ["file_sha256", "pixel_sha256", "dhash256"]:
        require(inventory[column].str.fullmatch(r"[0-9a-f]{64}").all(), f"Missing/invalid {column}")
    for column in ["width", "height"]:
        require(inventory[column].astype(str).str.fullmatch(r"[1-9]\d*").all(), "Invalid recorded dimensions")
    require(inventory.pixel_sha256.nunique() == expected["unique_pixels"], "Unexpected unique pixel count")
    duplicated = inventory[inventory.pixel_sha256.duplicated(keep=False)]
    pixel_components = {frozenset(frame.image) for _, frame in duplicated.groupby("pixel_sha256")}
    require(len(pixel_components) == expected["conflicting_groups"], "Unexpected internal duplicate group count")
    conflict_groups = groups[flags(groups.label_conflict)]
    require(len(conflict_groups) == expected["conflicting_groups"] and len(groups) == len(conflict_groups),
            "Expected exactly 27 conflicting groups and no other internal duplicate groups")
    require(flags(members.label_conflict).all(), "Member conflict flags disagree")
    require(set(members.group_id) == set(conflict_groups.group_id), "Group/member IDs disagree")
    require(len(members) == expected["quarantined_records"] and set(members.image) == set(duplicated.image),
            "Unexpected conflicting original membership/count")
    reference = inventory.set_index("image").loc[members.image].reset_index()
    require(members[list(inventory.columns)].reset_index(drop=True).equals(reference[list(inventory.columns)]),
            "Duplicate members differ from inventory audit fields")
    reported_components = set()
    for row in conflict_groups.itertuples(index=False):
        group = members[members.group_id.eq(row.group_id)]
        image_ids = str(row.image_ids).split(" | ")
        require(len(image_ids) == len(set(image_ids)) == len(group) == int(row.image_count) == 2,
                f"Invalid pair size: {row.group_id}")
        require(set(image_ids) == set(group.image), f"Summary/member image IDs disagree: {row.group_id}")
        require(set(str(row.original_labels).split(" | ")) == set(group.original_labels),
                f"Group original label summary disagrees: {row.group_id}")
        require(group.pixel_sha256.nunique() == 1 and group.original_labels.nunique() == 2,
                f"Group is not an exact conflicting-label pair: {row.group_id}")
        candidate_count = int(group.image.isin(candidates.image).sum())
        require(candidate_count == int(row.candidate_count) == 1 and len(group)-candidate_count == int(row.excluded_count) == 1,
                f"Candidate/excluded group counts disagree: {row.group_id}")
        reported_components.add(frozenset(group.image))
    require(reported_components == pixel_components, "Reported groups do not cover exact duplicate components")
    group_map = members.set_index("image").group_id
    quarantined_ids = set(members.image)

    def annotate(frame, candidate_manifest=False):
        if candidate_manifest:
            frame = frame.rename(columns={"review_status": "original_review_status"}).merge(
                inventory.rename(columns={"status": "original_audit_status"}),
                on=["image", "proposed_label"], how="left", validate="one_to_one", sort=False)
        else:
            frame = frame.rename(columns={"status": "original_audit_status"})
            frame["labels"] = frame.original_labels
            frame["original_review_status"] = frame.image.map(candidates.set_index("image").review_status).fillna("")
        frame["group_id"] = frame.image.map(group_map).fillna("")
        frame["is_single_label_candidate"] = frame.image.isin(candidates.image)
        conflict = frame.image.isin(quarantined_ids)
        frame["status"] = conflict.map({True: QUARANTINE, False: PENDING})
        frame["review_status"] = frame.status
        frame["quarantine_reason"] = conflict.map({True: REASON, False: ""})
        frame["reason_code"] = frame.quarantine_reason
        frame["eligible_for_training"] = False
        frame["eligible_for_unseen_evaluation"] = False
        frame["mapping_status"] = frame.labels.map(
            lambda label: "provisional_frogeye_to_black_rot_not_diagnostically_confirmed" if label == "frog_eye_leaf_spot"
            else "proposed_source_label_mapping" if label in MAPPING else "excluded_original_label_no_mapping")
        return frame

    all_candidates = annotate(candidates, candidate_manifest=True)
    removed = all_candidates[all_candidates.image.isin(quarantined_ids)].copy()
    remaining = all_candidates[~all_candidates.image.isin(quarantined_ids)].copy()
    quarantine = annotate(inventory[inventory.image.isin(quarantined_ids)].copy()).sort_values(
        ["group_id", "image"], kind="stable").reset_index(drop=True)
    require(len(removed) == expected["quarantined_candidates"], "Unexpected removed candidate count")
    require(len(remaining) == expected["remaining_candidates"], "Unexpected remaining candidate count")
    require(not set(remaining.image) & quarantined_ids, "Quarantine leaked into remaining candidates")
    require(set(removed.image) | set(remaining.image) == set(candidates.image), "Candidate partition incomplete")
    require(counts(candidates) == dict(sorted((Counter(removed.labels)+Counter(remaining.labels)).items())),
            "Candidate class counts do not reconcile")
    for frame in [quarantine, removed, remaining]:
        require(frame.image.is_unique and frame.labels.eq(frame.image.map(original)).all(), "Output labels/IDs changed")
    require(quarantine.group_id.ne("").all() and quarantine.quarantine_reason.eq(REASON).all(), "Missing quarantine provenance")
    summary = {**expected, "affected_group_ids": sorted(conflict_groups.group_id),
               "reason_codes": [REASON], "original_candidate_class_counts": counts(candidates),
               "quarantined_candidate_class_counts": counts(removed), "remaining_candidate_class_counts": counts(remaining),
               "remaining_candidate_counts_by_proposed_label": {str(k): int(v) for k, v in sorted(Counter(remaining.proposed_label).items())},
               "audit_status": "internal_quarantine_passed_cross_dataset_audit_pending",
               "all_remaining_candidates_approved_for_training": False,
               "frogeye_mapping": "provisional only; no diagnostic label assigned by quarantine",
               "pixel_hash_scheme": "sha256(UTF8(f'{oriented_width}x{oriented_height}|') + EXIF-oriented RGB bytes)",
               "dhash_scheme": "256 row-major right>left bits; RGB->L; 17x16 LANCZOS; 64 hex digits, big-endian",
               "cross_dataset_hash_compatibility": "PP2020/PV tuple-size pixel header differs; bridge required before pixel matching",
               "images_decoded_or_hashed": 0, "training_performed": False, "splits_assigned": False}
    frames = dict(zip(OUTPUTS[:3], [quarantine, removed, remaining]))
    return frames, summary


def protection_snapshot(project, output):
    """Hash metadata/weights; stat images without rereading their contents or decoding."""
    project, output = Path(project).resolve(), Path(output).resolve()
    protected, image_stats = {}, {}
    roots = [project / name for name in ["data", "artifacts", "results", "lesion_manifests"]]
    files = set()
    for root in roots:
        if root.exists():
            files.update(path for path in root.rglob("*") if path.is_file() and output not in path.resolve().parents)
    files.update(path for path in project.iterdir() if path.suffix.lower() in {".pt", ".pth", ".ckpt"})
    # PV source images are outside the repository; include their frozen source paths.
    for name in ["train", "validation", "test"]:
        manifest = project / f"artifacts/efficientnet_b0/{name}_manifest.csv"
        if manifest.is_file():
            files.update(Path(value) for value in pd.read_csv(manifest, usecols=["path"]).path)
    for path in sorted(files):
        resolved = str(path.resolve())
        if path.suffix.lower() in {".csv", ".json", ".pt", ".pth", ".ckpt"}:
            protected[resolved] = sha256(path)
        elif path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".zip"}:
            info = path.stat()
            image_stats[resolved] = [info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    return {"metadata_and_checkpoint_sha256": protected, "image_and_archive_stat": image_stats}


def run(project, output=None, expected=None):
    project = Path(project).resolve()
    audit = project / "data/field_manifests/pp2021_audit"
    output = Path(output).resolve() if output else audit / "internal_quarantine_v1"
    require(output.parent == audit and bool(re.fullmatch(r"internal_quarantine_v[1-9]\d*", output.name)),
            "Output must be an explicitly versioned pp2021_audit/internal_quarantine_vN directory")
    require(not output.is_symlink(), "Output symlink is not permitted")
    sources = {key: project / value for key, value in SOURCES.items()}
    input_hashes = {key: {"path": str(path), "sha256": sha256(path)} for key, path in sources.items()}
    before = protection_snapshot(project, output)
    tables = {key: pd.read_csv(path, dtype=str, keep_default_na=False) for key, path in sources.items()}
    frames, report = build_outputs(tables, expected)
    images = project / "data/plant_pathology_2021/train_images"
    require(all((images / image).is_file() for image in tables["inventory"].image), "A recorded PP2021 source image is missing")
    # Preserve older partial root outputs as evidence; never merge/overwrite their schemas.
    legacy = {str(path): sha256(path) for name in OUTPUTS if (path := audit / name).is_file()}
    report.update({"script_sha256": sha256(__file__), "source_inputs": input_hashes, "output_directory": str(output),
                   "preserved_preexisting_root_outputs": legacy,
                   "integrity": {"passed": True, "protected_metadata_and_checkpoint_files": len(before["metadata_and_checkpoint_sha256"]),
                                 "protected_image_and_archive_paths": len(before["image_and_archive_stat"]),
                                 "metadata_and_checkpoint_fingerprint": hashlib.sha256(json_bytes(before["metadata_and_checkpoint_sha256"])).hexdigest(),
                                 "image_and_archive_stat_fingerprint": hashlib.sha256(json_bytes(before["image_and_archive_stat"])).hexdigest(),
                                 "image_preservation_method": "read-only metadata transformation; before/after path,size,mtime_ns,ctime_ns; no image bytes reread",
                                 "limitation": "Stat comparison is not a fresh byte-level re-audit of source images; reuse existing inventory SHA-256",
                                 "source_labels_and_manifest_hashes_unchanged": True,
                                 "no_quarantined_ID_in_remaining_candidates": True,
                                 "class_counts_reconciled": True}})
    contents = {name: frame.to_csv(index=False, lineterminator="\n").encode("utf-8") for name, frame in frames.items()}
    report["output_csv_sha256"] = {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}
    contents[OUTPUTS[3]] = json_bytes(report)
    if output.exists():
        require(output.is_dir() and {path.name for path in output.iterdir()} == set(OUTPUTS),
                "Existing audit is incomplete/different; preserve it and select a new --output-dir internal_quarantine_vN")
        for name, content in contents.items():
            require((output / name).read_bytes() == content,
                    f"Existing output differs: {name}; preserve it and select a new --output-dir internal_quarantine_vN")
        mode = "verified_and_reused_identical_audit"
    else:
        require(before == protection_snapshot(project, output), "Protected inputs changed before publication; stopping")
        # Build together then publish a single new directory; only this validated output is writable.
        with tempfile.TemporaryDirectory(dir=audit, prefix=".internal_quarantine_stage_") as temporary:
            stage = Path(temporary).resolve()
            require(stage.parent == output.parent == audit and audit in stage.parents and audit in output.parents,
                    "Staging/publication paths escaped the audit directory")
            for name, content in contents.items():
                (stage / name).write_bytes(content)
            require(not output.exists(), "Output appeared during build; refusing overwrite")
            stage.rename(output)
        mode = "published_new_version"
    require(before == protection_snapshot(project, output), "Protected inputs changed; stop and disregard this audit")
    for key, path in sources.items():
        require(sha256(path) == input_hashes[key]["sha256"], f"Source changed: {path}")
    print(json.dumps({"mode": mode, "output_directory": str(output), "conflicting_groups": report["conflicting_groups"],
                      "quarantined_records": report["quarantined_records"], "quarantined_candidates": report["quarantined_candidates"],
                      "remaining_candidates": report["remaining_candidates"], "remaining_class_counts": report["remaining_candidate_class_counts"],
                      "integrity_passed": True, "cross_dataset_audit": "pending; no candidates authorized for training",
                      "images_decoded_or_hashed": 0, "training_performed": False, "splits_assigned": False}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
