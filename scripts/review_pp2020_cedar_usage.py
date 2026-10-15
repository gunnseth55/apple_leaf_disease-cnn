"""Review recorded Cedar use without running inference or changing dataset splits."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import pandas as pd

from audit_pp2020_for_field_training import inspect_image, sha256

ID_RE = re.compile(r"(?<![A-Za-z0-9_])Train_\d+(?![A-Za-z0-9_])")
IDENTITY_COLUMNS = {"image_id", "image_path", "path", "source_path", "filename", "relative_path", "image"}
TEXT_EXTENSIONS = {".py", ".ps1", ".md", ".json", ".csv", ".txt", ".log", ".ipynb"}


def csv_role(path, columns):
    name = path.name.lower()
    if any(word in name for word in ["excluded", "duplicate", "candidate", "audit"]):
        return "reference_only"
    if {"predicted_class", "predicted_label", "prediction", "predictions"} & set(columns):
        return "evaluated"
    if "train" in name:
        completed = (path.parent / "training_history.csv").is_file() and bool(list(path.parent.glob("*.pt")))
        return "trained" if completed else "training_manifest_membership_unconfirmed"
    if "val" in name:
        completed = (path.parent / "training_history.csv").is_file() and bool(list(path.parent.glob("*.pt")))
        return "model_selection_membership" if completed else "validation_manifest_membership_unconfirmed"
    if "test" in name or "evaluation" in path.as_posix().lower() or "metrics" in name:
        return "evaluation_membership_unconfirmed"
    return "manifest_membership_unconfirmed"


def summarize_roles(roles):
    if "trained" in roles or "model_selection_membership" in roles:
        return "confirmed_development_use", "training_or_validation_or_existing_benchmark;not_fresh_independent_test_for_prior_development"
    if "evaluated" in roles:
        return "confirmed_evaluation_use", "training_or_validation_or_repeat_existing_evaluation;not_fresh_independent_test_for_prior_development"
    if any("unconfirmed" in role for role in roles):
        return "prior_use_requires_review", "hold_split_assignment_until_membership_use_resolved"
    return "copy_verified_no_other_use_found_in_reviewed_records", "eligible_for_future_train_or_validation_or_test_subject_to_duplicate_group_split_and_final_protocol;assign_one_role_only"


def resolve_record_image(options):
    """Prefer available image content over a Git LFS pointer placeholder."""
    for candidate in options:
        if candidate.is_file():
            with candidate.open("rb") as stream:
                pointer = stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")
            if not pointer:
                return candidate
    return None


def relevant_files(project, output):
    files = []
    for root, directories, names in os.walk(project):
        directory = Path(root)
        directories[:] = [name for name in directories if name not in {".git", ".venv", "__pycache__", "node_modules"}
                          and (directory / name).resolve() != output.resolve()
                          and not (directory == project / "data" and name in {"field_manifests", "plant_pathology_2020", "lesion_zenodo"})]
        for name in names:
            path = directory / name
            if path.suffix.lower() in TEXT_EXTENSIONS:
                files.append(path)
    return sorted(files)


def review(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty review output directory")
    root = project / "data/plant_pathology_2020"
    if output == root or root in output.parents:
        raise ValueError("Review output must be outside the source dataset")
    labels_path = root / "extracted/train.csv"
    labels = pd.read_csv(labels_path)
    rust = labels[(labels.rust == 1) & (labels.healthy == 0) & (labels.scab == 0) & (labels.multiple_diseases == 0)]
    if len(rust) != 622 or rust.image_id.duplicated().any():
        raise ValueError("Expected exactly 622 uniquely labelled, single-disease rust images")
    audit_path = project / "data/field_manifests/pp2020_audit/pp2020_inventory.csv"
    inventory = pd.read_csv(audit_path).set_index("image_id")
    cedar_dir = project / "data/external_classification/cedar_pp2020"
    records, evidence, file_index, errors = {}, [], [], []
    cache, digest_to_ids, pixels_to_ids = {}, defaultdict(set), defaultdict(set)
    for image_id in rust.image_id:
        original = Path(inventory.loc[image_id, "image_path"])
        copies = [p for p in cedar_dir.glob(image_id + ".*") if p.is_file()]
        if len(copies) != 1:
            raise ValueError(f"No unique Cedar copy for {image_id}")
        original_sha, copy_sha = sha256(original), sha256(copies[0])
        if original_sha != copy_sha or original_sha != inventory.loc[image_id, "sha256"]:
            raise ValueError(f"Source/copy/audit hash mismatch for {image_id}")
        pixel_sha = inventory.loc[image_id, "pixel_sha256"]
        digest_to_ids[original_sha].add(image_id)
        pixels_to_ids[pixel_sha].add(image_id)
        records[image_id] = dict(image_id=image_id, source_image_path=str(original), copied_image_path=str(copies[0]),
                                 sha256=original_sha, roles=set(), evidence_files={"prepare_cedar_external.py", str(labels_path), str(copies[0])})
        cache[str(original.resolve())] = dict(sha256=original_sha, pixel_sha256=pixel_sha, image_status="ok")
        cache[str(copies[0].resolve())] = cache[str(original.resolve())]
        evidence.append(dict(image_id=image_id, role="copied", evidence_file=str(copies[0]),
                             locator="Source/copy SHA-256 equality; prepare_cedar_external.py copies bytes only", match="raw_sha256"))
    files = relevant_files(project, output)
    for path in files:
        relative = path.relative_to(project).as_posix()
        # Derived review/audit code and fixtures are not historical model-use evidence.
        self_audit = path.name in {Path(__file__).name, "audit_pp2020_for_field_training.py", "test_pp2020_audit.py", "test_cedar_usage_review.py"}
        try:
            text = path.read_text(encoding="utf-8-sig")
            references = [i for i, line in enumerate(text.splitlines(), 1) if "cedar_pp2020" in line.lower()]
            file_index.append(dict(path=relative, sha256=sha256(path), cedar_reference_lines=";".join(map(str, references)),
                                   scope="audit_not_usage" if self_audit else "reviewed_project_metadata"))
            if self_audit:
                continue
            if path.suffix.lower() != ".csv":
                # Textual occurrences are leads requiring review, never proof of training.
                # Extraction code and the pre-audit metadata snapshot are inspected separately.
                ids = set(ID_RE.findall(text)) & records.keys()
                if path.name != "cedar_checkpoint_metadata_review.json":
                    for image_id in ids:
                        records[image_id]["roles"].add("text_reference_unconfirmed")
                        records[image_id]["evidence_files"].add(relative)
                        evidence.append(dict(image_id=image_id, role="text_reference_unconfirmed",
                                             evidence_file=relative, locator="Text identity occurrence; review context", match="recorded_id"))
                continue
            frame = pd.read_csv(path, dtype=str).fillna("")
            columns = set(frame.columns) & IDENTITY_COLUMNS
            if not columns:
                continue
            role = csv_role(path, frame.columns)
            if role == "reference_only":
                continue
            for row_index, row in frame.iterrows():
                found = set()
                methods = set()
                for column in columns:
                    value = row[column]
                    ids = set(ID_RE.findall(value)) & records.keys()
                    if ids:
                        found.update(ids)
                        methods.add("recorded_id")
                    image = Path(value)
                    if column == "image_id" or image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                        continue
                    options = [image] if image.is_absolute() else [
                        project / image, path.parent / image,
                        project / "data/external_classification/kashmir" / image,
                        project / "data/external_classification/kashmir_recovered" / image,
                        project / "data/plant_pathology_2020/extracted/images" / image,
                        cedar_dir / image,
                    ]
                    # Kashmir's original download retains Git LFS pointer files;
                    # use the separately verified recovered image when available.
                    actual = resolve_record_image(options)
                    if actual is None:
                        # Filenames alone are not complete paths; full missing paths are logged.
                        if image.is_absolute() or len(image.parts) > 1:
                            errors.append(dict(file=relative, row=int(row_index)+2, error="Referenced image unavailable", path=value))
                        continue
                    key = str(actual.resolve())
                    if key not in cache:
                        cache[key] = inspect_image(actual)
                    info = cache[key]
                    if info["image_status"] != "ok":
                        errors.append(dict(file=relative, row=int(row_index)+2, error="Referenced image not decodable", path=value))
                    matches = digest_to_ids.get(info["sha256"], set()) | pixels_to_ids.get(info["pixel_sha256"], set())
                    if matches:
                        found.update(matches)
                        methods.add("raw_or_normalized_pixel_sha256")
                for image_id in found:
                    records[image_id]["roles"].add(role)
                    records[image_id]["evidence_files"].add(relative)
                    evidence.append(dict(image_id=image_id, role=role, evidence_file=relative,
                                         locator=f"CSV row {int(row_index)+2}", match=";".join(sorted(methods))))
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            errors.append(dict(file=relative, error=str(error)))
        if len(file_index) % 50 == 0:
            print(f"Reviewed metadata: {len(file_index)}/{len(files)}; image identities checked: {len(cache)}", flush=True)
    # Inspect history without printing unrelated shell commands or credentials.
    history = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/PowerShell/PSReadLine/ConsoleHost_history.txt"
    history_review = dict(available=history.is_file(), relevant_command_lines=[], limitations="Only persisted local PowerShell history; not a complete execution history")
    if history.is_file():
        for number, line in enumerate(history.read_text(encoding="utf-8-sig").splitlines(), 1):
            if "cedar_pp2020" in line.lower() or "prepare_cedar_external" in line.lower():
                # Record location only; commands are supporting leads, not proof of completion.
                history_review["relevant_command_lines"].append(number)
    git_result = subprocess.run(["git", "log", "--all", "--format=%H %s", "--", "*cedar*", "*Cedar*"],
                                cwd=project, capture_output=True, text=True)
    git_review = dict(returncode=git_result.returncode, cedar_file_commits=git_result.stdout.splitlines())
    if git_result.returncode:
        errors.append(dict(file="git history", error=git_result.stderr))
    historical = []
    commits = subprocess.run(["git", "log", "--all", "--format=%H", "-G", r"cedar_pp2020|Train_[0-9]+",
                              "--", "*.csv", "*.json", "*.log", "*.ipynb"], cwd=project, capture_output=True, text=True)
    if commits.returncode:
        errors.append(dict(file="git metadata history", error=commits.stderr))
    for commit in dict.fromkeys(commits.stdout.splitlines()):
        changed = subprocess.run(["git", "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit],
                                 cwd=project, capture_output=True, text=True)
        for filename in changed.stdout.splitlines():
            if Path(filename).suffix.lower() not in {".csv", ".json", ".log", ".ipynb"}:
                continue
            blob = subprocess.run(["git", "show", f"{commit}:{filename}"], cwd=project, capture_output=True, text=True)
            if blob.returncode:
                errors.append(dict(file=f"git:{commit}:{filename}", error="Historical blob unavailable"))
                continue
            ids = set(ID_RE.findall(blob.stdout)) & records.keys()
            # Saved content hashes also catch references with renamed filenames.
            content_ids = set().union(*(digest_to_ids[digest] for digest in digest_to_ids if digest in blob.stdout))
            ids.update(content_ids)
            historical.append(dict(commit=commit, path=filename, sha256=hashlib.sha256(blob.stdout.encode()).hexdigest(),
                                   cedar_ids=";".join(sorted(ids))))
            for image_id in ids:
                records[image_id]["roles"].add("historical_reference_unconfirmed")
                records[image_id]["evidence_files"].add(f"git:{commit}:{filename}")
                evidence.append(dict(image_id=image_id, role="historical_reference_unconfirmed",
                                     evidence_file=f"git:{commit}:{filename}", locator="Historical identity/content reference", match="id_or_sha256"))
    # Read existing local checkpoint metadata only; do not build or run any model.
    import torch
    checkpoints = []
    for checkpoint in sorted((project / "artifacts").rglob("*.pt")):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = {k: v for k, v in payload.items() if k != "model_state_dict"}
        serialized = json.dumps(metadata, default=str)
        ids = set(ID_RE.findall(serialized)) & records.keys()
        if "cedar_pp2020" in serialized.lower():
            ids.update(records)
        checkpoints.append(dict(path=str(checkpoint), sha256=sha256(checkpoint), metadata=metadata))
        for image_id in ids:
            records[image_id]["roles"].add("checkpoint_dataset_reference_unconfirmed")
            records[image_id]["evidence_files"].add(str(checkpoint))
            evidence.append(dict(image_id=image_id, role="checkpoint_dataset_reference_unconfirmed",
                                 evidence_file=str(checkpoint), locator="Checkpoint metadata reference", match="id_or_dataset_path"))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(historical).to_csv(output / "historical_metadata_review.csv", index=False)
    (output / "checkpoint_metadata_review.json").write_text(json.dumps(checkpoints, indent=2, default=str)+"\n", encoding="utf-8")
    pd.DataFrame(evidence).to_csv(output / "cedar_usage_evidence.csv", index=False)
    pd.DataFrame(file_index).to_csv(output / "reviewed_files.csv", index=False)
    rows = []
    for image_id, record in records.items():
        roles = record.pop("roles")
        status, allowed = summarize_roles(roles)
        if errors:
            status, allowed = "review_coverage_incomplete", "hold_split_assignment_until_coverage_errors_resolved"
        record.update(prior_role=";".join(["copied", *sorted(roles)]),
                      evidence_file=";".join(sorted(record.pop("evidence_files") | {str(output / "reviewed_files.csv"), str(output / "review_summary.json")})),
                      status=status, allowed_future_uses=allowed,
                      prior_evaluation="recorded" if "evaluated" in roles else "not_found_in_reviewed_records",
                      prior_training="recorded" if "trained" in roles else "not_found_in_reviewed_records",
                      prior_model_selection="recorded" if "model_selection_membership" in roles else "not_found_in_reviewed_records",
                      review_scope="Local project CSV/JSON/logs/notebooks/scripts, current checkpoint metadata snapshot, persisted shell history and Cedar file git history; no claim about deleted/external records",
                      identity_verified=True)
        rows.append(record)
    frame = pd.DataFrame(rows).sort_values("image_id", key=lambda s: s.str.extract(r"(\d+)", expand=False).astype(int))
    frame.to_csv(output / "pp2020_cedar_usage_review.csv", index=False)
    summary = dict(records=len(frame), status_counts=frame.status.value_counts().to_dict(),
                   roles_counts=frame.prior_role.value_counts().to_dict(), copied_source_hashes_verified=622,
                   metadata_files_reviewed=len(file_index), referenced_images_checked=len(cache), coverage_errors=errors,
                   history=history_review, git=git_review,
                   historical_metadata_blobs_reviewed=len(historical),
                   checkpoint_records_reviewed=len(checkpoints),
                   checkpoint_metadata_snapshot=str(output / "checkpoint_metadata_review.json"),
                   freshness_caveat="No use record found is not proof of never used. Copying alone is not contamination. A future test set must be locked before further development and disjoint from all training/validation and duplicate groups.",
                   audit_registry_unchanged=True, splits_assigned=False, inference_performed=False)
    (output / "review_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({**{k: v for k, v in summary.items() if k != "coverage_errors"},
                      "coverage_error_count": len(errors)}, indent=2))
    return frame, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    review(args.project, args.output_dir or args.project / "data/field_manifests/pp2020_cedar_usage_review_verified")


if __name__ == "__main__":
    main()
