"""Task D: build or verify versioned PP2020 field manifests. Never trains a model."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import tempfile

import pandas as pd

from appleleaf.config import APPLE_CLASSES
from audit_pp2020_for_field_training import sha256
from preserve_pp2020_heldout import group_permissions, validate_proposed_roles
from adjudicate_pp2020_duplicates import enforce_split_constraints

ALGORITHM = "hamilton_class_targets_forced_train_then_seeded_group_deficit_v1"
SPLITS = ("train", "val", "test")
BOOL_COLUMNS = ["source_record", "reserved_existing_evaluation", "previously_seen_by_segmenter",
                "blocked_by_adjudication", "eligible_classification_training", "eligible_new_pipeline_validation",
                "eligible_new_unseen_pipeline_test", "eligible_existing_evaluation", "cedar_task_a_clearance_used"]
PERMISSIONS = ["reserved_existing_evaluation", "previously_seen_by_segmenter", "blocked_by_adjudication",
               "proposed_classification_role", "eligible_classification_training", "eligible_new_pipeline_validation",
               "eligible_new_unseen_pipeline_test", "eligible_existing_evaluation"]
OUTPUT_CSVS = ("train.csv", "val.csv", "test.csv", "excluded.csv")
MANIFEST_COLUMNS = ["image_id", "image_path", "path", "group_id", "source", "class_name", "label",
                    "sha256", "pixel_sha256", "previously_seen_by_segmenter", "split"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    frame = pd.read_csv(path, keep_default_na=False)
    for column in frame:
        if column in BOOL_COLUMNS or column.startswith("same_split_required"):
            values = frame[column].astype(str).str.lower()
            require(values.isin(["true", "false"]).all(), f"Missing/invalid boolean {column} in {path}")
            frame[column] = values.eq("true")
    return frame


def numeric_id(value):
    require(bool(re.fullmatch(r"Train_\d+", str(value))), f"Invalid original image ID: {value}")
    return int(str(value).split("_")[1])


def targets(count):
    """Integer 70/15/15 Hamilton allocation; ties prefer train, then test, then val."""
    numerators = {"train": 70 * count, "val": 15 * count, "test": 15 * count}
    result = {name: value // 100 for name, value in numerators.items()}
    tie_order = {"train": 0, "test": 1, "val": 2}
    ordered = sorted(SPLITS, key=lambda name: (-(numerators[name] % 100), tie_order[name]))
    for name in ordered[:count - sum(result.values())]:
        result[name] += 1
    return result


def allocate_groups(canonical, seed):
    """Keep groups intact; charge mandatory training rows against full-class quotas."""
    rng, assignment, class_targets = random.Random(seed), {}, {}
    for label in sorted(canonical.label.unique()):
        frame = canonical[canonical.label == label]
        quota = targets(len(frame))
        class_targets[str(label)] = quota
        counts = dict.fromkeys(SPLITS, 0)
        grouped = []
        for group_id, rows in frame.groupby("group_id", sort=True):
            require(rows.label.nunique() == 1, f"Mixed-label accepted group: {group_id}")
            forced = bool(rows.previously_seen_by_segmenter.any())
            grouped.append((group_id, len(rows), forced))
        free = []
        for group_id, weight, forced in grouped:
            if forced:
                assignment[group_id] = "train"
                counts["train"] += weight
            else:
                free.append((group_id, weight))
        rng.shuffle(free)
        free.sort(key=lambda item: -item[1])  # Stable sort retains seeded order for ties.
        for group_id, weight in free:
            costs = {name: (counts[name] + weight - quota[name])**2 - (counts[name] - quota[name])**2
                     for name in SPLITS}
            best = min(costs.values())
            selected = rng.choice([name for name in SPLITS if costs[name] == best])
            assignment[group_id] = selected
            counts[selected] += weight
    return assignment, class_targets


def check_hash_chain(summary_path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for value, digest in summary["input_sha256"].items():
        path = Path(value)
        require(path.is_file() and sha256(path) == digest, f"Upstream audit provenance changed: {path}")


def load_inputs(project):
    base = project / "data/field_manifests"
    paths = {
        "cedar": base / "pp2020_cedar_usage_review_verified/pp2020_cedar_usage_review.csv",
        "groups": base / "pp2020_duplicate_adjudication/pp2020_split_group_constraints.csv",
        "members": base / "pp2020_duplicate_adjudication/pp2020_duplicate_group_members.csv",
        "roles": base / "pp2020_role_preservation/pp2020_classification_role_registry.csv",
        "file_roles": base / "pp2020_role_preservation/pp2020_all_file_role_registry.csv",
        "inventory": base / "pp2020_audit/pp2020_inventory.csv",
        "source_labels": project / "data/plant_pathology_2020/extracted/train.csv",
        "heldout_manifest": project / "data/healthy_structural_benchmark/manifest.csv",
        "segmentation_training_manifest": project / "lesion_manifests/train.csv",
        "task_b_summary": base / "pp2020_duplicate_adjudication/adjudication_summary.json",
        "task_c_summary": base / "pp2020_role_preservation/preservation_summary.json",
    }
    for path in paths.values():
        require(path.is_file(), f"Required input missing: {path}")
    hashes = {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}
    check_hash_chain(paths["task_b_summary"])
    check_hash_chain(paths["task_c_summary"])
    data = {name: read_csv(path) for name, path in paths.items() if path.suffix == ".csv"}
    inv, roles, groups, members, files, cedar = (data[name] for name in ["inventory", "roles", "groups", "members", "file_roles", "cedar"])
    require(len(inv) == 1821 and inv.image_id.nunique() == 1821, "Audited inventory must contain 1,821 unique IDs")
    require(roles.image_id.is_unique and set(roles.image_id) == set(inv.image_id), "Classification role registry has missing/extra/duplicate original IDs")
    require(groups.group_id.is_unique, "Duplicate constraint group IDs")
    require(members.image_path.is_unique and files.image_path.is_unique, "Duplicate file paths in member registries")
    require(set(members.image_path) == set(files.image_path), "All-file roles do not cover confirmed copies")
    require(set(groups.group_id) == set(members.group_id), "Duplicate constraints/member group mismatch")
    require(not groups.assigned_split.astype(str).str.len().any(), "Upstream groups already have split assignments")
    require(not roles.assigned_classification_split.astype(str).str.len().any(), "Classification roles already have split assignments")
    for frame in [roles, files, groups, members]:
        columns = [column for column in frame if column.startswith("same_split_required")]
        require(bool(columns) and all(frame[column].all() for column in columns), "Missing/contradictory duplicate group indivisibility permission")
    # Explode the group constraint's original ID list, then join on original IDs.
    mapping = groups.assign(image_id=groups.image_ids.str.split(";")).explode("image_id")
    require(mapping.image_id.is_unique and set(mapping.image_id) == set(inv.image_id), "Group constraints do not uniquely cover every original ID")
    constraints = mapping[["image_id", "group_id", "status", "split_constraint", "cedar_task_a_clearance_used"]].rename(
        columns={name: f"constraint_{name}" for name in ["group_id", "status", "split_constraint", "cedar_task_a_clearance_used"]})
    joined = roles.merge(constraints, on="image_id", how="left", validate="one_to_one")
    for column in ["group_id", "status", "split_constraint", "cedar_task_a_clearance_used"]:
        require(joined[column].eq(joined[f"constraint_{column}"]).all(), f"Contradictory group/role {column}")
    require(joined.source_record.all() and joined.member_role.eq("labelled_source").all(), "Role registry includes extracted aliases as originals")
    require(joined.image_path.is_unique, "Repeated canonical source paths")
    lookup = inv.set_index("image_id")
    labels = data["source_labels"].set_index("image_id")
    require(labels.index.is_unique and set(labels.index) == set(inv.image_id), "Source labels do not match the audited inventory")
    for row in joined.itertuples():
        original = lookup.loc[row.image_id]
        require(row.image_path == original.image_path and row.sha256 == original.sha256,
                f"Role/source image identity mismatch: {row.image_id}")
        for flag in ["healthy", "multiple_diseases", "rust", "scab"]:
            require(float(original[flag]) == float(labels.loc[row.image_id, flag]), f"Changed source label: {row.image_id}")
        if row.eligible_classification_training:
            require(original.label_status == "valid_single_label" and original.image_status == "ok", f"Invalid accepted source: {row.image_id}")
            value = float(row.label)
            require(value.is_integer() and 0 <= value < len(APPLE_CLASSES), f"Invalid class index: {row.image_id}")
            require(row.label_name == APPLE_CLASSES[int(value)] and row.label_name == original.label_name,
                    f"Four-class mapping mismatch: {row.image_id}")
            require(int(value) in {0, 2, 3}, "PP2020 must not supply unverified black-rot labels")
    file_lookup = files.set_index("image_path")
    group_lookup = groups.set_index("group_id")
    role_lookup = joined.set_index("image_id")
    for group_id, group in members.groupby("group_id", sort=False):
        expected = group_permissions(group)
        constraint = group_lookup.loc[group_id]
        require(len(group) == int(constraint.member_files), f"Wrong member count: {group_id}")
        require(set(group.status) == {constraint.status}, f"Group status mismatch: {group_id}")
        require(set(group[group.source_record].image_id) == set(constraint.image_ids.split(";")), f"Original ID membership mismatch: {group_id}")
        for record in group.to_dict("records"):
            recorded = file_lookup.loc[record["image_path"]]
            require(recorded.group_id == group_id and recorded.member_role == record["member_role"], "Copy path/group/role mismatch")
            for name in PERMISSIONS:
                require(recorded[name] == expected[name], f"Missing/contradictory file permission {name}: {record['image_id']}")
            if record["source_record"]:
                canonical_role = role_lookup.loc[record["image_id"]]
                for name in PERMISSIONS:
                    require(canonical_role[name] == expected[name], f"Missing/contradictory source permission {name}: {record['image_id']}")
    require(cedar.image_id.is_unique, "Duplicate Cedar review IDs")
    rust_ids = set(joined[joined.label_name.eq(APPLE_CLASSES[2])].image_id)
    require(len(cedar) == 622 and set(cedar.image_id) == rust_ids, "Missing/extra Cedar usage permissions")
    cedar_fields = cedar[["image_id", "status", "prior_role", "allowed_future_uses", "sha256", "identity_verified",
                          "prior_evaluation", "prior_training", "prior_model_selection"]].rename(
        columns={column: "cedar_"+column for column in cedar.columns if column != "image_id"})
    joined = joined.merge(cedar_fields, on="image_id", how="left", validate="one_to_one")
    for row in joined[joined.image_id.isin(rust_ids)].itertuples():
        require(row.cedar_status == "copy_verified_no_other_use_found_in_reviewed_records" and row.cedar_prior_role == "copied",
                f"Cedar prior-use permission requires review: {row.image_id}")
        require(row.cedar_sha256 == row.sha256 and str(row.cedar_identity_verified).lower() == "true", f"Cedar identity mismatch: {row.image_id}")
        require(row.cedar_allowed_future_uses == "eligible_for_future_train_or_validation_or_test_subject_to_duplicate_group_split_and_final_protocol;assign_one_role_only",
                f"Contradictory Cedar future use: {row.image_id}")
        require(all(getattr(row, name) == "not_found_in_reviewed_records" for name in ["cedar_prior_evaluation", "cedar_prior_training", "cedar_prior_model_selection"]),
                f"Cedar review contradicts fresh-use permission: {row.image_id}")
        require(row.cedar_task_a_clearance_used, f"Cedar clearance not inherited by duplicate group: {row.image_id}")
    # Mandatory historical roles: match live manifests by paths, not only IDs.
    for name, source, role, expected in [("heldout_manifest", "evaluation_only", "evaluation_only", 50),
                                         ("segmentation_training_manifest", "hard_negative", "segmentation_training", 40)]:
        selected = data[name][data[name].source.eq(source)]
        wanted = {str((Path(value) if Path(value).is_absolute() else project / value).resolve()) for value in selected.image_path}
        require(len(wanted) == expected and wanted == set(members[members.member_role.eq(role)].image_path), f"Changed historical role manifest: {name}")
    require(set(joined[joined.status.eq("quarantine_label_conflict")].image_id) == {"Train_379", "Train_1173"}, "Known conflict quarantine changed")
    # Recheck every referenced path and every canonical byte hash before freezing.
    require(all(Path(path).is_file() for path in members.image_path), "An original or confirmed copy path is missing")
    for row in joined.itertuples():
        require(sha256(row.image_path) == row.sha256, f"Source content changed: {row.image_id}")
    return joined, members, files, groups, inv, paths, hashes


def build_frames(joined, members, inventory, seed):
    eligible = joined[joined.eligible_classification_training].copy()
    require(not eligible.empty, "No accepted canonical images")
    require(eligible.groupby("group_id").label_name.nunique().le(1).all(), "Mixed-label accepted duplicate group")
    # Confirmed duplicate component = one original photograph. Use its lowest
    # original numeric ID, never an extracted copy, as the canonical source.
    eligible["_id_order"] = eligible.image_id.map(numeric_id)
    canonical = eligible.sort_values("_id_order").drop_duplicates("group_id").copy()
    canonical["label"] = canonical.label.astype(float).astype(int)
    assignment, quotas = allocate_groups(canonical, seed)
    source_lookup = inventory.set_index("image_id")
    rows = []
    for record in canonical.to_dict("records"):
        rows.append(dict(image_id=record["image_id"], image_path=record["image_path"], path=record["image_path"],
                         group_id=record["group_id"], source="plant_pathology_2020", class_name=record["label_name"],
                         label=int(record["label"]), sha256=record["sha256"], pixel_sha256=source_lookup.loc[record["image_id"], "pixel_sha256"],
                         previously_seen_by_segmenter=record["previously_seen_by_segmenter"], split=assignment[record["group_id"]]))
    accepted = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values("image_id", key=lambda values: values.map(numeric_id))
    selected_paths = set(accepted.image_path)
    role_lookup = joined.set_index("image_id")
    excluded = []
    for member in members.to_dict("records"):
        if member["image_path"] in selected_paths:
            continue
        original = role_lookup.loc[member["image_id"]]
        if not member["source_record"]:
            reason = "extracted_copy_alias_not_an_independent_sample"
            if original.reserved_existing_evaluation:
                reason += ";reserved_evaluation_group"
            if original.blocked_by_adjudication:
                reason += ";"+original.status
        elif original.reserved_existing_evaluation:
            reason = "reserved_evaluation_group_outside_all_new_splits"
        elif original.blocked_by_adjudication:
            reason = original.status
        elif member["group_id"] in assignment:
            reason = "duplicate_noncanonical_original_same_photograph"
        else:
            raise ValueError(f"Accepted source has no assignment or exclusion reason: {member['image_id']}")
        excluded.append(dict(image_id=member["image_id"], image_path=member["image_path"], group_id=member["group_id"],
                             source="plant_pathology_2020", source_record=member["source_record"], member_role=member["member_role"],
                             class_name=original.label_name, label=original.label, split="excluded", exclusion_reason=reason,
                             inherited_group_split=assignment.get(member["group_id"], "")))
    frames = {name+".csv": accepted[accepted.split.eq(name)].reset_index(drop=True) for name in SPLITS}
    frames["excluded.csv"] = pd.DataFrame(excluded)
    return frames, quotas


def validate_frames(frames, joined, members, file_roles, groups, inventory):
    accepted = pd.concat([frames[name+".csv"] for name in SPLITS], ignore_index=True)
    excluded = frames["excluded.csv"]
    eligible = joined[joined.eligible_classification_training]
    require(accepted.image_id.is_unique and accepted.image_path.is_unique, "A canonical source appears more than once")
    require(accepted.group_id.is_unique, "A confirmed photograph has multiple canonical rows")
    require(set(accepted.group_id) == set(eligible.group_id), "Every accepted photograph must have exactly one split")
    require(not accepted.groupby("group_id").split.nunique().gt(1).any(), "Duplicate groups cross splits")
    require(accepted.pixel_sha256.ne("").all() and accepted.pixel_sha256.is_unique, "Repeated/missing pixel identity in accepted splits")
    require(accepted.sha256.is_unique, "Repeated source bytes in accepted splits")
    require(set(accepted.image_path).isdisjoint(set(excluded.image_path)), "Accepted/excluded overlap")
    require(set(accepted.image_path) | set(excluded.image_path) == set(members.image_path), "All file memberships must reconcile")
    require(excluded.image_path.is_unique, "Repeated exclusions")
    original_excluded = excluded[excluded.source_record]
    require(len(accepted)+len(original_excluded) == len(inventory), "Original inventory counts do not reconcile")
    require(set(accepted.image_id) | set(original_excluded.image_id) == set(inventory.image_id), "Original IDs do not reconcile")
    protected = set(joined[joined.reserved_existing_evaluation | joined.blocked_by_adjudication].group_id)
    require(not set(accepted.group_id) & protected, "Reserved/quarantined records entered new splits")
    historical = set(joined[joined.previously_seen_by_segmenter].group_id)
    require(len(historical) == 40, "Expected 40 segmentation-training groups")
    require(set(accepted[accepted.group_id.isin(historical)].split) == {"train"}, "Previously trained segmenter image entered unseen evaluation")
    require(historical <= set(frames["train.csv"].group_id), "An accepted mandatory training image was omitted")
    require(len(set(joined[joined.reserved_existing_evaluation].group_id)) == 50, "Expected 50 reserved groups")
    for name in SPLITS:
        require(frames[name+".csv"].split.eq(name).all(), f"Incorrect split column in {name}")
    for row in accepted.itertuples():
        require(Path(row.image_path).is_file() and row.path == row.image_path, f"Missing/noncanonical image path: {row.image_id}")
        require(float(row.label).is_integer() and APPLE_CLASSES[int(row.label)] == row.class_name and int(row.label) in {0, 2, 3}, "Invalid four-class mapping")
    plan = accepted[["group_id", "split"]].rename(columns={"split": "role"})
    validate_proposed_roles(plan, file_roles)
    enforce_split_constraints(accepted[["group_id", "split"]], members)
    inherited = excluded.group_id.map(accepted.set_index("group_id").split).fillna("")
    require(excluded.inherited_group_split.fillna("").eq(inherited).all(), "Extracted copy failed to inherit its group assignment")
    return dict(every_accepted_photograph_assigned_once=True, no_confirmed_group_crosses_splits=True,
                reserved_and_quarantined_groups_outside_all_new_splits=True, all_40_previously_trained_groups_train_only=True,
                copies_do_not_inflate_counts=True, original_inventory_reconciles=True, all_file_members_reconcile=True,
                paths_exist=True, four_class_mapping_valid=True, aliases_inherit_group_assignment=True,
                prior_role_validators_passed=True)


def run(project, output, seed=42, verify=False):
    project, output = Path(project).resolve(), Path(output).resolve()
    require(bool(re.fullmatch(r"pp2020_split_v[1-9]\d*", output.name)), "Output folder must have an explicit version such as pp2020_split_v1")
    require(output.parent == project / "data/field_manifests", "Versioned splits must be under data/field_manifests")
    require(0 <= seed < 2**32, "Seed must be a nonnegative 32-bit integer")
    if not verify:
        require(not output.exists(), "Frozen split version already exists; use --verify or create an explicitly new version")
    else:
        require(output.is_dir(), f"Frozen split version does not exist: {output}")
    joined, members, file_roles, groups, inventory, input_paths, input_hashes = load_inputs(project)
    if verify:
        report = json.loads((output / "split_report.json").read_text(encoding="utf-8"))
        integrity = json.loads((output / "split_integrity_report.json").read_text(encoding="utf-8"))
        require(report["version"] == output.name and report["algorithm"] == ALGORITHM, "Unknown frozen version/algorithm")
        require(report["class_to_index"] == dict(zip(APPLE_CLASSES, range(4))), "Frozen class mapping changed")
        require(report["source_inputs"] == input_hashes, "Audit inputs differ from the frozen experiment")
        require(sha256(output / "split_report.json") == integrity["split_report_sha256"], "Frozen split report changed")
        for filename in OUTPUT_CSVS:
            require(sha256(output / filename) == report["manifest_sha256"][filename] == integrity["manifest_sha256"][filename], f"Frozen manifest changed: {filename}")
        frames = {name: read_csv(output / name) for name in OUTPUT_CSVS}
        checks = validate_frames(frames, joined, members, file_roles, groups, inventory)
        expected, _ = build_frames(joined, members, inventory, int(report["seed"]))
        for filename in OUTPUT_CSVS:
            require(frames[filename].to_csv(index=False, lineterminator="\n") == expected[filename].to_csv(index=False, lineterminator="\n"), f"Assignment not reproducible: {filename}")
        require(integrity["passed"] and checks == integrity["checks"], "Integrity checks disagree with the frozen report")
        print(f"PASS: {output.name} verified; seed {report['seed']}; no files rewritten.")
        return report
    frames, quotas = build_frames(joined, members, inventory, seed)
    checks = validate_frames(frames, joined, members, file_roles, groups, inventory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=output.name+"_building_") as temporary:
        stage = Path(temporary)
        for filename, frame in frames.items():
            frame.to_csv(stage / filename, index=False, lineterminator="\n")
        manifest_hashes = {filename: sha256(stage / filename) for filename in OUTPUT_CSVS}
        counts = {name: {"total": len(frames[name+".csv"]), "by_class": frames[name+".csv"].class_name.value_counts().to_dict()}
                  for name in SPLITS}
        report = dict(version=output.name, seed=seed, algorithm=ALGORITHM,
                      algorithm_details="Integer 70/15/15 class targets by largest remainder (tie order train,test,val). Prior segmenter-training groups are allocated first against train quotas. Remaining groups: seeded shuffle, stable size-descending ordering, minimize incremental squared class-quota error; seeded ties. One canonical original per confirmed photograph.",
                      created_at_utc=datetime.now(timezone.utc).isoformat(), builder_sha256=sha256(Path(__file__)),
                      class_to_index=dict(zip(APPLE_CLASSES, range(4))), source_inputs=input_hashes,
                      manifest_sha256=manifest_hashes, class_targets=quotas, counts=counts,
                      original_inventory_records=len(inventory), accepted_canonical_photographs=sum(counts[name]["total"] for name in SPLITS),
                      excluded_original_records=int(frames["excluded.csv"].source_record.sum()),
                      excluded_copy_aliases=int((~frames["excluded.csv"].source_record).sum()),
                      protected_evaluation_groups=50, forced_training_groups=40,
                      split_changes_require_new_version=True, training_performed=False,
                      limitations="PP2020 has no black-rot supervision. Task A clearance is bounded to recorded local history. Frozen test must not guide training, threshold tuning or checkpoint selection.")
        (stage / "split_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        integrity = dict(passed=True, checks=checks, manifest_sha256=manifest_hashes,
                         split_report_sha256=sha256(stage / "split_report.json"),
                         input_images_verified_against_audit=True, source_images_modified=False, training_performed=False)
        (stage / "split_integrity_report.json").write_text(json.dumps(integrity, indent=2)+"\n", encoding="utf-8")
        for name, path in input_paths.items():
            require(sha256(path) == input_hashes[name]["sha256"], f"Input changed during split building: {path}")
        require(not output.exists(), "Split version appeared during build; refusing overwrite")
        stage.rename(output)  # Publish the fully validated version together.
    print(json.dumps({"version": output.name, "counts": counts, "integrity_passed": True,
                      "excluded_original_records": report["excluded_original_records"],
                      "excluded_copy_aliases": report["excluded_copy_aliases"], "training_performed": False}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true", help="Verify an existing frozen version without rewriting it")
    args = parser.parse_args(argv)
    run(args.project, args.output_dir or args.project / "data/field_manifests/pp2020_split_v1", args.seed, args.verify)


if __name__ == "__main__":
    main()
