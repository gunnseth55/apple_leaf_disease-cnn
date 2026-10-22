"""Save a verified compact C2-C snapshot and group-aware visual-review queue."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from verify_pp2021_cross_dataset import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/field_manifests/pp2021_cross_dataset_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/2026-10-23_pp2021_cross_dataset"))
    args = parser.parse_args()
    verification = verify(args.root)
    pairs = pd.read_csv(args.root / "perceptual_candidates.csv", keep_default_na=False)
    members = pd.read_csv(args.root / "confirmed_group_members.csv", keep_default_na=False).set_index("node_id")
    policy = pd.read_csv(args.root / "group_restrictions.csv", keep_default_na=False).set_index("group_id")
    pp = pd.read_csv(args.root / "pp2021_overlap_status.csv", keep_default_na=False).set_index("node_id")
    pairs["reference_confirmed_group"] = pairs.right_id.map(members.group_id)
    pairs["effective_reference_roles"] = pairs.reference_confirmed_group.map(policy.roles)
    pairs["reference_protected_history"] = pairs.reference_confirmed_group.map(policy.protected_history)
    pairs["reference_classifier_train_seen"] = pairs.reference_confirmed_group.map(policy.classifier_training_seen)
    pairs["reference_segmenter_train_seen"] = pairs.reference_confirmed_group.map(policy.segmenter_training_seen)
    pairs["pp2021_c2b_status"] = pairs.left_id.map(pp.c2b_status)
    pairs["remaining_c2b_candidate"] = pairs.pp2021_c2b_status.eq("pending_cross_dataset_audit")
    pairs["review_priority"] = pairs.apply(lambda row: 1 if row.reference_protected_history else
                                           2 if row.reference_classifier_train_seen or row.reference_segmenter_train_seen else 3, axis=1)
    pairs["review_priority_reason"] = pairs.review_priority.map({1: "possible_protected_validation_or_evaluation_overlap",
                                                               2: "possible_prior_training_overlap",
                                                               3: "other_unresolved_perceptual_candidate"})
    pairs = pairs.sort_values(["review_priority", "remaining_c2b_candidate", "dhash_distance", "pair_id"],
                              ascending=[True, False, True, True])
    sheets = pd.read_csv(args.root / "review_sheet_index.csv", keep_default_na=False)
    pairs = pairs.merge(sheets, on="pair_id", how="left", validate="one_to_one")
    summary = dict(pairs=len(pairs), unique_pp2021_records=int(pairs.left_id.nunique()),
                   remaining_candidates_with_review_hold=int(pairs[pairs.remaining_c2b_candidate].left_id.nunique()),
                   remaining_candidate_possible_protected_overlaps=int(pairs[pairs.remaining_c2b_candidate & pairs.reference_protected_history].left_id.nunique()),
                   pair_counts_by_priority={str(k): int(v) for k,v in pairs.review_priority.value_counts().sort_index().items()},
                   dhash_distance_pair_counts={str(k): int(v) for k,v in pairs.dhash_distance.value_counts().sort_index().items()},
                   all_decisions_unresolved=True, automatic_rejections=0, training_approvals=0,
                   note="Priority uses confirmed reference-group history, including copy aliases; it is not a visual duplicate decision")
    contents = {name: (args.root / name).read_bytes() for name in
                ["eligibility_summary.json", "cross_dataset_integrity_report.json", "hash_compatibility_report.json"]}
    contents["verification_report.json"] = (json.dumps(verification, indent=2)+"\n").encode()
    contents["review_priority_summary.json"] = (json.dumps(summary, indent=2)+"\n").encode()
    contents["priority_review_queue.csv"] = pairs.to_csv(index=False, lineterminator="\n").encode()
    contents["snapshot_sha256.json"] = (json.dumps({k: hashlib.sha256(v).hexdigest() for k,v in contents.items()}, indent=2)+"\n").encode()
    if args.output_dir.exists():
        for name, content in contents.items():
            if not (args.output_dir / name).is_file() or (args.output_dir / name).read_bytes() != content:
                raise ValueError("Existing snapshot differs; choose a new --output-dir")
        print("Existing identical summary snapshot verified; no files rewritten")
    else:
        args.output_dir.mkdir(parents=True)
        for name, content in contents.items():
            (args.output_dir / name).write_bytes(content)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
