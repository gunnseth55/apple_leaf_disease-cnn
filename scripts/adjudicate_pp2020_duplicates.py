"""Adjudicate saved PP2020 duplicate links and export split-indivisible groups."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from audit_pp2020_for_field_training import FLAGS, inspect_image, sha256


def judge_pair(left, right, decision=None):
    if left["image_status"] != "ok" or right["image_status"] != "ok":
        return "unresolved_integrity", "Cannot confirm identity without complete decoding"
    if left["pixel_sha256"] and left["pixel_sha256"] == right["pixel_sha256"]:
        return "confirmed_duplicate", "EXIF-oriented RGB dimensions and pixel SHA-256 match"
    distance = (int(left["dhash256"], 16) ^ int(right["dhash256"], 16)).bit_count()
    if distance > 4:
        return "false_positive", "Different pixels and recomputed dHash distance exceeds 4"
    if decision:
        if decision.get("decision") not in {"confirmed_duplicate", "false_positive"} or not decision.get("reason") or not decision.get("reviewer"):
            raise ValueError("Visual decision requires confirmed_duplicate/false_positive, reason and reviewer")
        return decision["decision"], f"Visual reviewer {decision['reviewer']}: {decision['reason']}"
    return "needs_visual_review", "dHash proximity alone is not proof of duplication"


class Components:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, node):
        while node != self.parent[node]:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def join(self, left, right):
        self.parent[self.find(right)] = self.find(left)


def restriction(signatures, roles, cedar_cleared=True, pending=False):
    if len(signatures) > 1:
        return "quarantine_label_conflict", "no_split_until_labels_adjudicated"
    if pending:
        return "hold_pending_pair_review", "no_split_until_pair_review_complete"
    if "evaluation_only" in roles and "segmentation_training" in roles:
        return "quarantine_prior_role_conflict", "no_split_until_prior_role_conflict_resolved"
    if "evaluation_only" in roles:
        return "reserved_evaluation", "evaluation_only_never_training_or_validation"
    if "segmentation_training" in roles:
        return "accepted_previously_trained", "training_only_for_independent_final_pipeline_evaluation"
    if not cedar_cleared:
        return "hold_cedar_use_review", "no_split_until_prior_use_resolved"
    signature = next(iter(signatures))
    if len(signature) != 4 or sum(signature) != 1 or signature[1] == 1:
        return "excluded_from_c1", "no_c1_split_invalid_or_multiple_disease_labels"
    return "accepted", "one_future_split_only_train_or_validation_or_test"


def enforce_split_constraints(assignments, members):
    """Validate a future node/group split plan; no split assignment is made here."""
    if not {"group_id", "split"}.issubset(assignments.columns):
        raise ValueError("Assignments require group_id and split")
    if set(assignments.group_id) - set(members.group_id):
        raise ValueError("Unknown duplicate group in split assignments")
    if set(assignments.split) - {"train", "validation", "val", "test", "evaluation"}:
        raise ValueError("Invalid split name")
    joined = members.merge(assignments, on="group_id", how="inner")
    if joined.groupby("group_id").split.nunique().gt(1).any():
        raise ValueError("Confirmed duplicate group assigned to multiple splits")
    for row in joined.itertuples():
        if row.status.startswith(("quarantine", "hold", "excluded")):
            raise ValueError(f"Ineligible group assigned a split: {row.group_id}")
        if row.status == "reserved_evaluation" and row.split not in {"test", "evaluation"}:
            raise ValueError("Reserved evaluation group cannot enter training/validation")
        if row.status == "accepted_previously_trained" and row.split != "train":
            raise ValueError("Previously trained group cannot enter independent evaluation")


def fonts(size):
    candidate = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(candidate), size) if candidate.is_file() else ImageFont.load_default()


def tile(pair, project, size=(1100, 285)):
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    title_font, font = fonts(18), fonts(15)
    draw.text((10, 4), f"{pair['pair_id']} | {pair['match_type']} | dHash {pair['distance']}/256 | {pair['decision']}", fill="black", font=title_font)
    for x, side in [(10, "left"), (565, "right")]:
        with Image.open(pair[f"{side}_path"]) as image:
            preview = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (275, 210))
        canvas.paste(preview, (x, 35))
        draw.text((x + 285, 42), pair[f"{side}_id"], fill="black", font=title_font)
        draw.text((x + 285, 72), pair[f"{side}_label"], fill="black", font=font)
        draw.text((x + 285, 100), pair[f"{side}_role"], fill="black", font=font)
    draw.text((10, 255), f"Pixel SHA equal: {pair['pixel_hash_equal']} | Byte SHA equal: {pair['raw_hash_equal']} | group {pair['group_id']}", fill="black", font=font)
    return canvas


def run(project, output, visual_decisions=None):
    project, output = Path(project).resolve(), Path(output).resolve()
    source_root = project / "data/plant_pathology_2020"
    if output == source_root or source_root in output.parents:
        raise ValueError("Outputs must be outside source images")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new/empty adjudication output folder")
    base = project / "data/field_manifests/pp2020_audit"
    inventory = pd.read_csv(base / "pp2020_inventory.csv").fillna("")
    registry = pd.read_csv(base / "previous_use_registry.csv").fillna("")
    relations = pd.read_csv(base / "duplicate_screening.csv").fillna("")
    labels_path = source_root / "extracted/train.csv"
    labels = pd.read_csv(labels_path).set_index("image_id")
    cedar_path = project / "data/field_manifests/pp2020_cedar_usage_review_verified/pp2020_cedar_usage_review.csv"
    cedar_review = pd.read_csv(cedar_path).set_index("image_id")
    nodes = {}
    def add_node(row, role):
        path = str(Path(row["image_path"]).resolve())
        image_id = row["image_id"]
        if image_id not in labels.index:
            raise ValueError(f"Unresolved node label: {path}")
        signature = tuple(int(labels.loc[image_id, flag]) for flag in FLAGS)
        label = next((flag for flag, value in zip(FLAGS, signature) if value == 1), "invalid")
        nodes[path] = dict(path=path, image_id=image_id, role=role, signature=signature,
                           label=label, expected_sha=row["sha256"], source_record=role == "labelled_source")
    for row in inventory.to_dict("records"):
        add_node(row, "labelled_source")
    for row in registry.to_dict("records"):
        add_node(row, row["purpose"])
    components = Components(nodes)
    decisions = {}
    if visual_decisions:
        review = pd.read_csv(visual_decisions).fillna("")
        if review.pair_id.duplicated().any():
            raise ValueError("Duplicate visual decision pair IDs")
        decisions = {row["pair_id"]: row for row in review.to_dict("records")}
    cache, pairs, pending_paths = {}, [], set()
    def current(path):
        if path not in cache:
            result = inspect_image(path)
            if result["sha256"] != nodes[path]["expected_sha"]:
                raise ValueError(f"Source changed since audit: {path}")
            cache[path] = result
        return cache[path]
    for index, relation in enumerate(relations.to_dict("records")):
        left = str(Path(inventory.iloc[int(relation["left_row"])].image_path).resolve())
        if relation["scope"] == "within_labelled":
            right = str(Path(inventory.iloc[int(float(relation["right_row"]))].image_path).resolve())
        else:
            right = str(Path(relation["right_path"]).resolve())
        if nodes[left]["image_id"] != relation["left_id"] or nodes[right]["image_id"] != relation["right_id"]:
            raise ValueError("Relation IDs do not match their image records")
        a, b = current(left), current(right)
        pair_id = f"PAIR_{index+1:04d}"
        decision, reason = judge_pair(a, b, decisions.get(pair_id))
        distance = (int(a["dhash256"], 16) ^ int(b["dhash256"], 16)).bit_count()
        pixel_equal = bool(a["pixel_sha256"]) and a["pixel_sha256"] == b["pixel_sha256"]
        pair = dict(pair_id=pair_id, scope=relation["scope"], left_id=nodes[left]["image_id"],
                    right_id=nodes[right]["image_id"], left_path=left, right_path=right,
                    left_label=nodes[left]["label"], right_label=nodes[right]["label"],
                    left_role=nodes[left]["role"], right_role=nodes[right]["role"], distance=distance,
                    match_type="exact_pixel_hash" if pixel_equal else "dhash_candidate",
                    pixel_hash_equal=pixel_equal, raw_hash_equal=a["sha256"] == b["sha256"],
                    decision=decision, reason=reason,
                    label_conflict=nodes[left]["signature"] != nodes[right]["signature"])
        pairs.append(pair)
        if decision == "confirmed_duplicate":
            components.join(left, right)
        elif decision != "false_positive":
            pending_paths.update([left, right])
        if (index + 1) % 100 == 0:
            print(f"Reverified {index+1}/{len(relations)} pairs", flush=True)
    if set(decisions) - {pair["pair_id"] for pair in pairs}:
        raise ValueError("Visual decisions include unknown pair IDs")
    groups = defaultdict(list)
    for path in nodes:
        groups[components.find(path)].append(path)
    membership, summaries, path_groups = [], [], {}
    for paths in groups.values():
        original_ids = sorted({nodes[path]["image_id"] for path in paths if nodes[path]["source_record"]}, key=lambda value: int(value.split("_")[1]))
        group_id = "PP2020_" + original_ids[0]
        roles = {nodes[path]["role"] for path in paths}
        signatures = {nodes[path]["signature"] for path in paths}
        cleared = True
        if "cedar_prior_use_unresolved" in roles:
            for image_id in original_ids:
                if image_id not in cedar_review.index:
                    cleared = False
                    continue
                review_row = cedar_review.loc[image_id]
                source = next(nodes[path] for path in paths if nodes[path]["source_record"] and nodes[path]["image_id"] == image_id)
                cleared &= review_row["status"] == "copy_verified_no_other_use_found_in_reviewed_records" and review_row["sha256"] == source["expected_sha"]
        status, constraint = restriction(signatures, roles, cleared, bool(set(paths) & pending_paths))
        summary = dict(group_id=group_id, member_files=len(paths), distinct_labelled_records=len(original_ids),
                       image_ids=";".join(original_ids), label_signatures=";".join(map(str, sorted(signatures))),
                       prior_roles=";".join(sorted(roles)), status=status, split_constraint=constraint,
                       same_split_required=True, assigned_split="", cedar_task_a_clearance_used="cedar_prior_use_unresolved" in roles and cleared)
        summaries.append(summary)
        for path in paths:
            path_groups[path] = group_id
            membership.append({**summary, "image_id": nodes[path]["image_id"], "image_path": path,
                               "member_role": nodes[path]["role"], "source_record": nodes[path]["source_record"]})
    for pair in pairs:
        pair["group_id"] = path_groups[pair["left_path"]]
    # Verify every confirmed link stays in one component; unconfirmed links never join.
    assert all(path_groups[pair["left_path"]] == path_groups[pair["right_path"]]
               for pair in pairs if pair["decision"] == "confirmed_duplicate")
    output.mkdir(parents=True, exist_ok=True)
    contact = output / "contact_sheets"
    contact.mkdir()
    display = sorted(pairs, key=lambda pair: (not pair["label_conflict"], pair["pair_id"]))
    pages = []
    for start in range(0, len(display), 12):
        batch = display[start:start+12]
        sheet = Image.new("RGB", (2200, 6*285), "#dddddd")
        page = contact / f"pairs_{start//12+1:03d}.jpg"
        for position, pair in enumerate(batch):
            sheet.paste(tile(pair, project), ((position % 2)*1100, (position//2)*285))
            pair["contact_sheet"] = str(page.relative_to(output))
            pair["tile_position"] = position+1
        sheet.save(page, quality=92)
        pages.append(page)
    for pair in pairs:
        if pair["label_conflict"]:
            tile(pair, project, (1100,285)).save(output / f"{pair['pair_id']}_label_conflict.png")
    pair_frame, member_frame, group_frame = pd.DataFrame(pairs), pd.DataFrame(membership), pd.DataFrame(summaries)
    pair_frame.to_csv(output / "pp2020_duplicate_adjudications.csv", index=False)
    pair_frame[pair_frame.match_type == "exact_pixel_hash"].to_csv(output / "exact_pixel_matches.csv", index=False)
    pair_frame[pair_frame.match_type == "dhash_candidate"].to_csv(output / "dhash_candidates.csv", index=False)
    pair_frame[pair_frame.decision == "false_positive"].to_csv(output / "false_positive_pairs.csv", index=False)
    member_frame.to_csv(output / "pp2020_duplicate_group_members.csv", index=False)
    group_frame.to_csv(output / "pp2020_split_group_constraints.csv", index=False)
    member_frame[member_frame.status.str.startswith("quarantine")].to_csv(output / "quarantined_group_members.csv", index=False)
    pd.DataFrame([dict(pair_id=pair["pair_id"], decision="", reason="", reviewer="") for pair in pairs if pair["decision"] == "needs_visual_review"],
                 columns=["pair_id", "decision", "reason", "reviewer"]).to_csv(output / "visual_decisions_template.csv", index=False)
    summary = dict(input_relations=len(pairs), decision_counts=dict(Counter(pair["decision"] for pair in pairs)),
                   exact_pixel_matches=int((pair_frame.match_type == "exact_pixel_hash").sum()),
                   dhash_only_candidates=int((pair_frame.match_type == "dhash_candidate").sum()),
                   labelled_source_copy_alias_relations=int((pair_frame.left_id == pair_frame.right_id).sum()),
                   cross_labelled_record_relations=int((pair_frame.left_id != pair_frame.right_id).sum()),
                   duplicate_components=int((group_frame.member_files > 1).sum()),
                   groups_with_multiple_labelled_records=int((group_frame.distinct_labelled_records > 1).sum()),
                   all_components=len(groups), member_files=len(nodes), source_records=len(inventory),
                   group_status_counts=group_frame.status.value_counts().to_dict(),
                   contact_sheet_pages=len(pages), source_images_modified=False, splits_assigned=False,
                   caveat="A confirmed copy alias is not an additional independent training example. Split only by group_id; quarantine overrides all permissions. Task A is bounded to available local history.",
                   input_sha256={str(p): sha256(p) for p in [labels_path, base / "pp2020_inventory.csv", base / "duplicate_screening.csv", cedar_path]})
    (output / "adjudication_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    links = "\n".join(f'<h2>Page {i+1}</h2><a href="{page.relative_to(output).as_posix()}"><img loading="lazy" style="max-width:100%" src="{page.relative_to(output).as_posix()}" /></a>' for i,page in enumerate(pages))
    (output / "contact_sheet_index.html").write_text('<!doctype html><meta charset="utf-8"><title>PP2020 duplicate pairs</title><h1>All duplicate relations</h1><p>Each pair shows IDs, original labels, roles, recomputed distance and decision. Conflicting labels appear first. Exact identity is verified by pixel hashes; thumbnails are review previews.</p>'+links, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return pair_frame, member_frame, group_frame, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--visual-decisions", type=Path)
    args = parser.parse_args()
    run(args.project, args.output_dir or args.project / "data/field_manifests/pp2020_duplicate_adjudication", args.visual_decisions)


if __name__ == "__main__":
    main()
