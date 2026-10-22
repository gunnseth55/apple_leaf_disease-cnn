"""Independent saved-output verification; reads hashes/roles, loads no model or images."""

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path

import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def truth(value):
    require(value in {"True", "False"}, "Invalid boolean in output")
    return value == "True"


class EightBlockIndex:
    """Independent partition: <=4 changed bits leave at least four of eight 32-bit blocks intact."""
    def __init__(self, values):
        self.buckets = defaultdict(set)
        for value in set(values):
            for i in range(8):
                self.buckets[i, (value >> (32*i)) & 0xffffffff].add(value)

    def query(self, value):
        votes = Counter()
        for i in range(8):
            votes.update(self.buckets.get((i, (value >> (32*i)) & 0xffffffff), ()))
        return {other for other, count in votes.items() if count >= 4 and (other^value).bit_count() <= 4}


def verify(root):
    root = Path(root).resolve()
    report = json.loads((root / "cross_dataset_integrity_report.json").read_text())
    require(report["complete"] and report["integrity_passed"], "Incomplete audit cannot be verified")
    files = {path.relative_to(root).as_posix(): digest(path) for path in root.rglob("*")
             if path.is_file() and path.name != "cross_dataset_integrity_report.json"}
    require(files == report["output_sha256"], "Output file coverage or hash mismatch")
    for path, expected in report["source_inputs"].items():
        require(digest(path) == expected, f"Source changed: {path}")
    registry = read(root / "reference_registry.csv")
    require(registry.node_id.is_unique and registry.path.is_unique, "Duplicate nodes/paths")
    nodes = registry.set_index("node_id").to_dict("index")
    query = registry[registry.dataset.eq("pp2021")]
    refs = registry[~registry.dataset.eq("pp2021")]
    require(len(query) == 18632 and len(refs) == 6958, "Unexpected photo coverage")
    receipts = read(root / "photo_freshness_receipts.csv")
    require(set(receipts.node_id) == set(nodes) and receipts.node_id.is_unique, "Freshness coverage differs")
    for row in receipts.itertuples(index=False):
        require(nodes[row.node_id]["raw_sha256"] == row.raw_sha256, "Raw SHA receipt mismatch")
        path = Path(row.path)
        stat = path.stat()
        require([stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] == [int(row.size), int(row.mtime_ns), int(row.ctime_ns)],
                f"Photo file metadata changed since matching: {path}")
    inputs = list(report["source_inputs"])
    inventory_path = next(path for path in inputs if path.replace("\\", "/").endswith("pp2021_audit/image_inventory.csv"))
    inventory = read(inventory_path).set_index("image")
    for row in query.itertuples(index=False):
        original = inventory.loc[row.original_id]
        require((row.raw_sha256, row.pixel_sha256, row.dhash256) ==
                (original.file_sha256, original.pixel_sha256, original.dhash256), "PP2021 saved identity not preserved")
        require(json.loads(row.original_labels) == [original.original_labels], "Original PP2021 label changed")
    # Rebuild confirmed groups independently using graph connectivity, not the audit's union-find.
    graph = {key: set() for key in nodes}
    raw_index, pixel_index, seeds = defaultdict(list), defaultdict(list), defaultdict(list)
    for key, row in nodes.items():
        raw_index[row["raw_sha256"]].append(key)
        pixel_index[row["pixel_sha256"]].append(key)
        for seed in json.loads(row["seeds"]):
            seeds[seed].append(key)
    for groups in [raw_index, pixel_index, seeds]:
        for ids in groups.values():
            for other in ids[1:]:
                graph[ids[0]].add(other)
                graph[other].add(ids[0])
    rebuilt = {}
    for key in sorted(nodes):
        if key in rebuilt:
            continue
        queue, group = deque([key]), set([key])
        while queue:
            for neighbor in graph[queue.popleft()]:
                if neighbor not in group:
                    group.add(neighbor)
                    queue.append(neighbor)
        group_id = "CROSS_"+hashlib.sha256("\n".join(sorted(group)).encode()).hexdigest()[:24]
        for member in group:
            rebuilt[member] = group_id
    memberships = read(root / "confirmed_group_members.csv")
    require(memberships.node_id.is_unique and memberships.set_index("node_id").group_id.to_dict() == rebuilt,
            "Confirmed groups include a missing or unsupported union")
    exact, near = read(root / "exact_matches.csv"), read(root / "perceptual_candidates.csv")
    def pair_key(left, right):
        return tuple(sorted([left, right]))
    def pairs(frame):
        keys = {pair_key(row.left_id, row.right_id) for row in frame.itertuples(index=False)}
        require(len(keys) == len(frame) and frame.pair_id.is_unique, "Duplicate pair records")
        return keys
    reported_exact, reported_near = pairs(exact), pairs(near)
    expected_exact = set()
    for row in query.itertuples(index=False):
        for other in set(raw_index[row.raw_sha256]) | set(pixel_index[row.pixel_sha256]):
            if other != row.node_id:
                expected_exact.add(pair_key(row.node_id, other))
    require(reported_exact == expected_exact, "Exact pair coverage differs from inverted raw/pixel identities")
    ref_hashes, pp_hashes = defaultdict(list), defaultdict(list)
    for row in refs.itertuples(index=False):
        ref_hashes[int(row.dhash256, 16)].append(row.node_id)
    for row in query.itertuples(index=False):
        pp_hashes[int(row.dhash256, 16)].append(row.node_id)
    ref_index, internal_index = EightBlockIndex(ref_hashes), EightBlockIndex(pp_hashes)
    expected_near = set()
    for value, left_ids in pp_hashes.items():
        candidate_ids = [key for match in ref_index.query(value) for key in ref_hashes[match]]
        candidate_ids += [key for match in internal_index.query(value) if match >= value for key in pp_hashes[match]]
        for left in left_ids:
            for right in candidate_ids:
                if left == right:
                    continue
                a, b = nodes[left], nodes[right]
                if a["raw_sha256"] != b["raw_sha256"] and a["pixel_sha256"] != b["pixel_sha256"]:
                    expected_near.add(pair_key(left, right))
    require(reported_near == expected_near, "Independent eight-block screen differs from five-block audit")
    pending = set()
    for row in near.itertuples(index=False):
        a, b = nodes[row.left_id], nodes[row.right_id]
        require((int(a["dhash256"], 16)^int(b["dhash256"], 16)).bit_count() == int(row.dhash_distance) <= 4,
                "Perceptual distance mismatch")
        confirmed = rebuilt[row.left_id] == rebuilt[row.right_id]
        require(row.status == ("previously_confirmed_component" if confirmed else "pending_visual_review"), "Pair review status mismatch")
        if not confirmed:
            pending.update([row.left_id, row.right_id])
    for row in exact.itertuples(index=False):
        a, b = nodes[row.left_id], nodes[row.right_id]
        require(truth(row.raw_match) == (a["raw_sha256"] == b["raw_sha256"]) and
                truth(row.pixel_match) == (a["pixel_sha256"] == b["pixel_sha256"]), "Exact flags mismatch")
    by_group = defaultdict(list)
    for key, group_id in rebuilt.items():
        by_group[group_id].append(key)
    policies = read(root / "group_restrictions.csv").set_index("group_id")
    require(set(policies.index) == set(by_group), "Restriction group coverage mismatch")
    for group_id, keys in by_group.items():
        roles = set().union(*(set(json.loads(nodes[key]["roles"])) for key in keys))
        families = set().union(*(set(json.loads(nodes[key]["label_families"])) for key in keys))
        protected = bool(roles & {"classifier_validation", "classifier_test", "segmenter_validation", "segmenter_test", "protected_evaluation", "external_evaluation"})
        trained = bool(roles & {"classifier_train", "segmenter_train"})
        conflict = any(truth(nodes[key]["sticky_quarantine"]) for key in keys) or (len(families) > 1 and not families <= {"frog_eye_leaf_spot", "black_rot"})
        policy = policies.loc[group_id]
        require(truth(policy.protected_history) == protected and truth(policy.prior_role_conflict) == (protected and trained), "Role restrictions lost")
        require(truth(policy.label_conflict) == conflict and truth(policy.review_pending) == bool(set(keys) & pending), "Conflict/review hold lost")
        require(not truth(policy.eligible_for_c2_training) and not truth(policy.eligible_for_unseen_evaluation), "Automatic approval found")
    status = read(root / "pp2021_overlap_status.csv")
    require(status.image.is_unique and set(status.image) == set(inventory.index), "Status universe is not all 18632 records")
    require(int(status.c2b_status.eq("quarantined_label_conflict").sum()) == 54, "Original quarantine not preserved")
    require(status[status.c2b_status.eq("quarantined_label_conflict")].status.eq("quarantined_label_conflict").all(), "Quarantine cleared")
    summary = json.loads((root / "eligibility_summary.json").read_text())
    candidates = status[status.c2b_status.eq("pending_cross_dataset_audit")]
    require(len(candidates) == 14464 and Counter(candidates.original_labels) ==
            Counter({"scab": 4826, "frog_eye_leaf_spot": 3181, "rust": 1833, "healthy": 4624}), "Candidate class reconciliation failed")
    require(summary["exact_pair_count"] == len(exact) and summary["nonexact_perceptual_pairs"] == len(near), "Summary pair totals differ")
    template = read(root / "review_decisions_template.csv")
    sheets = read(root / "review_sheet_index.csv")
    pending_pair_ids = set(near[near.status.eq("pending_visual_review")].pair_id)
    require(set(template.pair_id) == set(sheets.pair_id) == pending_pair_ids, "Review evidence coverage incomplete")
    require(template.decision.eq("unresolved").all() and template.reviewer.eq("").all(), "Unexpected new visual adjudication")
    require(all((root / path).is_file() for path in sheets.sheet), "Missing review sheet")
    return dict(passed=True, photo_nodes=len(nodes), pp2021_records=len(query), reference_paths=len(refs),
                confirmed_groups=len(by_group), exact_pairs=len(exact), nonexact_perceptual_pairs=len(near),
                pending_visual_review_pairs=len(pending_pair_ids), all_original_quarantine_records_preserved=54,
                independent_index="eight 32-bit blocks, >=4 matching blocks; exact pair sets equal main five-block screen",
                source_and_output_hashes_verified=True, photo_file_metadata_unchanged=True,
                unsupported_confirmed_unions=0, training_performed=False, model_or_image_loading_performed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/field_manifests/pp2021_cross_dataset_v1"))
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
