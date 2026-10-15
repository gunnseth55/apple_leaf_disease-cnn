"""Task C: export pipeline-aware classification permissions, without assigning splits."""

import argparse
from pathlib import Path
import json

import pandas as pd

from audit_pp2020_for_field_training import sha256


def group_permissions(group):
    """Prior roles apply to the entire confirmed component, including renamed copies."""
    roles = set(group.member_role)
    statuses = set(group.status)
    if len(statuses) != 1:
        raise ValueError("Inconsistent adjudicated status within a duplicate group")
    status = next(iter(statuses))
    heldout, trained = "evaluation_only" in roles, "segmentation_training" in roles
    blocked = status.startswith(("quarantine", "hold", "excluded")) or (heldout and trained)
    if blocked:
        proposed = "quarantine_or_excluded_no_new_split"
        reason = status if not (heldout and trained) else "conflicting_training_and_heldout_history"
    elif heldout:
        proposed = "reserved_existing_evaluation_only"
        reason = "Preserve the earlier structural-negative benchmark and all confirmed copies"
    elif trained:
        proposed = "classification_training_only_or_unused"
        reason = "Eligible for classifier training; segmenter already saw this group, so it is not unseen to the pipeline"
    else:
        proposed = "unassigned_single_future_split"
        reason = "Eligible under Task A/B evidence; allocate the whole component to one role"
    return dict(reserved_existing_evaluation=heldout, previously_seen_by_segmenter=trained,
                blocked_by_adjudication=blocked, proposed_classification_role=proposed,
                eligible_classification_training=not blocked and not heldout,
                eligible_new_pipeline_validation=not blocked and not heldout and not trained,
                eligible_new_unseen_pipeline_test=not blocked and not heldout and not trained,
                eligible_existing_evaluation=heldout and not blocked,
                same_split_required=True, permission_reason=reason)


def validate_proposed_roles(plan, registry):
    """Check future group- or file-based plans, including aliases; never allocate splits."""
    if "role" not in plan:
        raise ValueError("Proposed plan needs a role column")
    proposed = plan.copy()
    if "image_path" in proposed:
        mapping = registry.set_index("image_path").group_id
        proposed["image_path"] = proposed.image_path.map(lambda value: str(Path(value).resolve()))
        if not proposed.image_path.isin(mapping.index).all():
            raise ValueError("Unknown image path: cannot bypass the prior-use registry")
        actual_groups = proposed.image_path.map(mapping)
        if "group_id" in proposed and not proposed.group_id.eq(actual_groups).all():
            raise ValueError("Image path and group ID disagree")
        proposed["group_id"] = actual_groups
    if "group_id" not in proposed:
        raise ValueError("Proposed plan needs group_id or image_path")
    groups = registry.drop_duplicates("group_id").set_index("group_id")
    if not proposed.group_id.isin(groups.index).all():
        raise ValueError("Unknown group ID")
    proposed["role"] = proposed.role.replace({"val": "validation"})
    active = proposed[proposed.role != "unused"]
    if active.groupby("group_id").role.nunique().gt(1).any():
        raise ValueError("Confirmed aliases assigned to different roles/splits")
    permissions = {"train": "eligible_classification_training", "validation": "eligible_new_pipeline_validation",
                   "test": "eligible_new_unseen_pipeline_test", "existing_evaluation": "eligible_existing_evaluation"}
    for row in proposed.itertuples():
        if row.role == "unused":
            continue
        if row.role not in permissions:
            raise ValueError(f"Unknown role: {row.role}")
        if not bool(groups.loc[row.group_id, permissions[row.role]]):
            raise ValueError(f"Forbidden {row.role} role for {row.group_id}: {groups.loc[row.group_id, 'permission_reason']}")


def run(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    dataset = project / "data/plant_pathology_2020"
    if output == dataset or dataset in output.parents:
        raise ValueError("Output must be outside source images")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new/empty output folder")
    base = project / "data/field_manifests"
    adjudication = base / "pp2020_duplicate_adjudication"
    members_path = adjudication / "pp2020_duplicate_group_members.csv"
    groups_path = adjudication / "pp2020_split_group_constraints.csv"
    inventory_path = base / "pp2020_audit/pp2020_inventory.csv"
    original_manifest_paths = [project / "data/healthy_structural_benchmark/manifest.csv", project / "lesion_manifests/train.csv"]
    inputs = [members_path, groups_path, inventory_path, *original_manifest_paths]
    before = {str(path): sha256(path) for path in inputs}
    members = pd.read_csv(members_path).fillna("")
    groups = pd.read_csv(groups_path).fillna("")
    inventory = pd.read_csv(inventory_path).fillna("")
    if members.image_path.duplicated().any() or groups.group_id.duplicated().any():
        raise ValueError("Duplicate membership paths/group rows")
    if set(members.group_id) != set(groups.group_id):
        raise ValueError("Group/membership mismatch")
    if set(members[members.source_record].image_id) != set(inventory.image_id):
        raise ValueError("Original labelled records are not fully covered")
    # Verify the live prior-use manifests still match the adjudicated reserved groups.
    for path, source, expected in [(original_manifest_paths[0], "evaluation_only", 50),
                                   (original_manifest_paths[1], "hard_negative", 40)]:
        manifest = pd.read_csv(path)
        selected = manifest[manifest.source == source]
        if len(selected) != expected:
            raise ValueError(f"Expected {expected} {source} images")
        actual = {str((Path(value) if Path(value).is_absolute() else project / value).resolve()) for value in selected.image_path}
        member_role = "evaluation_only" if source == "evaluation_only" else "segmentation_training"
        recorded = set(members[members.member_role == member_role].image_path)
        if actual != recorded:
            raise ValueError(f"Previous-use manifest changed since adjudication: {path}")
    policies = []
    for group_id, group in members.groupby("group_id", sort=False):
        policy = group_permissions(group)
        recorded = groups.set_index("group_id").loc[group_id]
        if recorded.status != group.status.iloc[0] or int(recorded.member_files) != len(group):
            raise ValueError(f"Group constraint changed: {group_id}")
        policies.append(dict(group_id=group_id, **policy))
    permissions = pd.DataFrame(policies)
    registry = members.merge(permissions, on="group_id", validate="many_to_one")
    registry["assigned_classification_split"] = ""  # This step only records permissions.
    originals = registry[registry.source_record].copy()
    originals = originals.merge(inventory[["image_id", "label_name", "label", "image_status", "sha256"]],
                                on="image_id", validate="one_to_one")
    if len(originals) != 1821:
        raise ValueError("Expected 1,821 canonical labelled sources")
    # Verify source identity again; do not write or copy image files.
    for row in originals.itertuples():
        if sha256(row.image_path) != row.sha256:
            raise ValueError(f"Source image changed since the audit: {row.image_id}")
    training = originals[originals.eligible_classification_training].copy()
    fresh = originals[originals.eligible_new_unseen_pipeline_test].copy()
    heldout = registry[registry.reserved_existing_evaluation].copy()
    seen = originals[originals.previously_seen_by_segmenter].copy()
    assert len(heldout[heldout.source_record]) == 50 and heldout.group_id.nunique() == 50
    assert len(seen) == 40 and seen.eligible_classification_training.all()
    assert not heldout.eligible_classification_training.any()
    assert not seen.eligible_new_unseen_pipeline_test.any()
    assert not seen.eligible_new_pipeline_validation.any()
    assert not set(training.group_id) & set(heldout.group_id)
    assert not set(fresh.group_id) & set(seen.group_id)
    # Exercise permission checks against the real protected image aliases.
    for path in heldout.image_path:
        try:
            validate_proposed_roles(pd.DataFrame([dict(image_path=path, role="train")]), registry)
        except ValueError:
            pass
        else:
            raise AssertionError("Held-out alias can enter training")
    validate_proposed_roles(pd.DataFrame([dict(group_id=row.group_id, role="train") for row in seen.itertuples()]), registry)
    if before != {str(path): sha256(path) for path in inputs}:
        raise RuntimeError("Input metadata changed during reconciliation")
    output.mkdir(parents=True, exist_ok=True)
    originals.to_csv(output / "pp2020_classification_role_registry.csv", index=False)
    registry.to_csv(output / "pp2020_all_file_role_registry.csv", index=False)
    heldout.to_csv(output / "protected_heldout_members.csv", index=False)
    seen.to_csv(output / "segmentation_training_classification_reconciliation.csv", index=False)
    training.to_csv(output / "classification_training_candidates.csv", index=False)
    fresh.to_csv(output / "new_pipeline_holdout_candidates.csv", index=False)
    summary = dict(canonical_labelled_records=len(originals), all_file_members=len(registry),
                   protected_evaluation_source_images=int(heldout.source_record.sum()),
                   protected_evaluation_groups=int(heldout.group_id.nunique()),
                   protected_evaluation_file_members=len(heldout),
                   confirmed_heldout_aliases=len(heldout)-int(heldout.source_record.sum()),
                   segmentation_training_sources=len(seen), segmentation_training_file_members=int(registry.previously_seen_by_segmenter.sum()),
                   previous_segmentation_training_allowed_classification_training=int(seen.eligible_classification_training.sum()),
                   training_candidate_sources=len(training),
                   training_candidate_class_counts=training.label_name.value_counts().to_dict(),
                   possible_new_unseen_pipeline_holdout_sources=len(fresh),
                   role_counts=originals.proposed_classification_role.value_counts().to_dict(),
                   splits_assigned=False, source_images_modified=False, input_manifests_unchanged=True,
                   input_sha256=before,
                   note="Candidate permissions are not split assignments. Existing evaluation images remain reserved and are not advertised as a fresh test set. Previously trained segmenter groups may train the classifier but cannot be unseen combined-pipeline validation/test.")
    (output / "preservation_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return registry, originals, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.project, args.output_dir or args.project / "data/field_manifests/pp2020_role_preservation")


if __name__ == "__main__":
    main()
