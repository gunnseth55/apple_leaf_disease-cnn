"""C2-C: read-only identity matching and review holds; never trains or assigns splits."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps, __version__ as PILLOW_VERSION

from quarantine_pp2021_conflicts import (
    SOURCES, OUTPUTS as QUARANTINE_OUTPUTS, build_outputs, json_bytes, require, sha256,
)


PIXEL_METHOD = "rgb_exif_wh_pipe_sha256_v1"
DHASH_METHOD = "rgb_exif_lanczos17x16_right_gt_left_rowmajor_msb256_v1"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASSES = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"]
FAMILIES = dict(zip(CLASSES, ["scab", "black_rot", "rust", "healthy"]))


def stat_signature(path):
    info = Path(path).stat()
    return [info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def compatible_hashes(path):
    """Reference-side bridge; no PP2021-wide image decoding."""
    with Image.open(path) as image:
        image.load()
        raw_rgb = image.convert("RGB")
        legacy_rgb = hashlib.sha256(str(raw_rgb.size).encode() + raw_rgb.tobytes()).hexdigest()
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        pixels = rgb.tobytes()
        pixel = hashlib.sha256(f"{rgb.width}x{rgb.height}|".encode() + pixels).hexdigest()
        legacy_oriented = hashlib.sha256(str(rgb.size).encode() + pixels).hexdigest()
        gray = np.asarray(rgb.convert("L").resize((17, 16), Image.Resampling.LANCZOS))
        dhash = np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes().hex()
    return dict(pixel_sha256=pixel, dhash256=dhash, width=rgb.width, height=rgb.height,
                legacy_rgb_tuple=legacy_rgb, legacy_exif_tuple=legacy_oriented,
                pixel_method=PIXEL_METHOD, dhash_method=DHASH_METHOD, pillow_version=PILLOW_VERSION)


class HammingIndex:
    """Exact radius-four search via five blocks; no pair truncation."""
    WIDTHS = (52, 51, 51, 51, 51)

    def __init__(self, values):
        self.values = set(values)
        self.buckets = defaultdict(set)
        self.comparisons = 0
        self.queries = 0
        for value in sorted(self.values):
            for key in self.keys(value):
                self.buckets[key].add(value)

    @classmethod
    def keys(cls, value):
        offset = 0
        for index, width in enumerate(cls.WIDTHS):
            yield index, (value >> offset) & ((1 << width)-1)
            offset += width

    def query(self, value):
        possible = set()
        for key in self.keys(value):
            possible.update(self.buckets.get(key, ()))
        self.comparisons += len(possible)
        self.queries += 1
        return [(other, distance) for other in sorted(possible) if (distance := (value ^ other).bit_count()) <= 4]

    def summary(self):
        return dict(unique_hashes=len(self.values), queries=self.queries, full_hamming_comparisons=self.comparisons,
                    largest_bucket=max(map(len, self.buckets.values()), default=0),
                    threshold=4, bit_blocks=list(self.WIDTHS), truncated=False)


class Components:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, node):
        while node != self.parent[node]:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def join(self, left, right):
        a, b = sorted([self.find(left), self.find(right)])
        self.parent[b] = a


class Registry:
    def __init__(self, project):
        self.project = Path(project).resolve()
        self.nodes, self.by_path = {}, {}
        self.evidence, self.auxiliary, self.inputs = [], [], {}
        self.coverage = {}

    def source(self, relative):
        path = Path(relative)
        path = path if path.is_absolute() else self.project / path
        path = path.resolve()
        require(path.is_file(), f"Required source missing: {path}")
        self.inputs[str(path)] = sha256(path)
        return path

    def csv(self, relative, expected=None):
        path = self.source(relative)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if expected is not None:
            require(len(frame) == expected, f"Unexpected count {len(frame)} vs {expected}: {path}")
        return frame, str(path)

    def photo(self, path, dataset, original_id, label, family, role, evidence, row,
              raw="", legacy="", legacy_method="", dhash="", seed="", sticky=False, mapping_uncertain=False):
        path = Path(path)
        path = (path if path.is_absolute() else self.project / path).resolve()
        require(path.is_file() and path.suffix.lower() in EXTENSIONS, f"Missing/invalid photo: {path}")
        key = str(path)
        node_id = self.by_path.get(key)
        if node_id is None:
            node_id = f"{dataset}:{original_id}:{hashlib.sha256(key.encode()).hexdigest()[:16]}"
            require(node_id not in self.nodes, "Node identity collision")
            self.by_path[key] = node_id
            self.nodes[node_id] = dict(node_id=node_id, dataset=dataset, original_id=str(original_id), path=key,
                                       original_labels=set(), label_families=set(), roles=set(), seeds=set(),
                                       expected_raw=set(), expected_legacy=set(), expected_dhash=set(),
                                       sticky_quarantine=False, mapping_uncertain=False, stat=stat_signature(path))
        node = self.nodes[node_id]
        if label:
            node["original_labels"].add(str(label))
        if family:
            node["label_families"].add(str(family))
        node["roles"].add(role)
        if raw:
            require(bool(re.fullmatch(r"[0-9a-f]{64}", raw)), f"Invalid raw SHA: {key}")
            node["expected_raw"].add(raw)
            require(len(node["expected_raw"]) == 1, f"Inconsistent saved raw identities: {key}")
        if legacy:
            require(legacy_method in {"legacy_rgb_tuple", "legacy_exif_tuple"}, "Unknown legacy pixel method")
            node["expected_legacy"].add((legacy_method, legacy))
        if dhash:
            require(bool(re.fullmatch(r"[0-9a-f]{64}", dhash)), "Invalid dHash256")
            node["expected_dhash"].add(dhash)
        if seed:
            node["seeds"].add(seed)
        node["sticky_quarantine"] |= sticky
        node["mapping_uncertain"] |= mapping_uncertain
        self.evidence.append(dict(node_id=node_id, role=role, source_file=evidence, source_row=row,
                                  original_label=label, seed_group=seed, evidence_status="recorded_membership"))
        return node

    def table(self):
        return pd.DataFrame([{key: json.dumps(sorted(value)) if isinstance(value, set) else value
                              for key, value in node.items() if key != "stat"} for node in self.nodes.values()])


def validate_quarantine(registry, directory):
    report_path = registry.source(directory / QUARANTINE_OUTPUTS[3])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report["integrity"]["passed"], "C2-B integrity failed")
    for entry in report["source_inputs"].values():
        require(sha256(registry.source(entry["path"])) == entry["sha256"], "C2-B source changed")
    for name, expected in report["output_csv_sha256"].items():
        require(sha256(registry.source(directory / name)) == expected, "C2-B output changed")
    tables = {key: registry.csv(relative)[0] for key, relative in SOURCES.items()}
    expected_frames, summary = build_outputs(tables)
    for name, frame in expected_frames.items():
        require(hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest() ==
                report["output_csv_sha256"][name], "C2-B transformation differs")
    return tables, summary


def completed_training(registry, folder, c1=False):
    history, _ = registry.csv(folder / "training_history.csv", 10 if c1 else 15)
    registry.source(folder / ("apple_efficientnet_b0_field_c1_checkpoint.pt" if c1 else
                             "apple_efficientnet_b0_robust_checkpoint.pt" if folder.name.endswith("robust") else
                             "apple_efficientnet_b0_checkpoint.pt"))
    config = json.loads(registry.source(folder / "run_config.json").read_text())
    require(set(history.epoch.astype(int)) == set(range(1, len(history)+1)), "Incomplete training history")
    if c1:
        summary = json.loads(registry.source(folder / "training_summary.json").read_text())
        require(summary["training_complete"] and summary["best_epoch"] == 10, "C1 training not complete")
        for path, expected in config["source_manifest_sha256"].items():
            require(sha256(registry.source(path)) == expected, "C1 frozen source changed")
        require(sha256(registry.source(config["initialization_checkpoint"])) == config["initialization_sha256"],
                "C1 initialization changed")


def build_registry(project, quarantine_dir):
    registry = Registry(project)
    tables, quarantine_summary = validate_quarantine(registry, quarantine_dir)
    inventory = tables["inventory"]
    pp_groups = tables["members"].set_index("image").group_id.to_dict()
    for i, row in enumerate(inventory.itertuples(index=False), 2):
        node = registry.photo(Path("data/plant_pathology_2021/train_images") / row.image, "pp2021", row.image,
                              row.original_labels, row.original_labels, "pp2021_candidate" if row.proposed_label else "excluded_original_label",
                              str(project / SOURCES["inventory"]), i, raw=row.file_sha256,
                              seed=f"pp2021:{pp_groups[row.image]}" if row.image in pp_groups else "",
                              sticky=row.image in pp_groups, mapping_uncertain=row.original_labels == "frog_eye_leaf_spot")
        node.update(pixel_sha256=row.pixel_sha256, dhash256=row.dhash256, width=int(row.width), height=int(row.height),
                    pixel_method=PIXEL_METHOD, dhash_method=DHASH_METHOD, proposed_label=row.proposed_label,
                    c2b_status="quarantined_label_conflict" if row.image in pp_groups else
                    "pending_cross_dataset_audit" if row.proposed_label else "excluded_original_label")
    registry.coverage["pp2021"] = len(inventory)
    # PV original source files, including aliases removed by exact deduplication.
    pv_folder = project / "artifacts/efficientnet_b0"
    completed_training(registry, pv_folder)
    completed_training(registry, project / "artifacts/efficientnet_b0_robust")
    c1 = project / "artifacts/efficientnet_b0_field_c1_v1"
    completed_training(registry, c1, c1=True)
    for split, count in [("train", 2214), ("validation", 475), ("test", 475)]:
        reference, path = registry.csv(pv_folder / f"{split}_manifest.csv", count)
        robust, _ = registry.csv(project / f"artifacts/efficientnet_b0_robust/{split}_manifest.csv", count)
        cols = ["path", "class_name", "label", "image_hash"]
        require(reference[cols].sort_values("path").reset_index(drop=True).equals(
            robust[cols].sort_values("path").reset_index(drop=True)), "Reference/robust PV inventory mismatch")
        require(reference.path.is_unique, "Duplicate PV split path")
        for i, row in enumerate(reference.itertuples(index=False), 2):
            require(CLASSES[int(row.label)] == row.class_name, "PV class mapping differs")
            registry.photo(row.path, "plantvillage", Path(row.path).name, row.class_name, FAMILIES[row.class_name],
                           "classifier_train" if split == "train" else f"classifier_{split}", path, i,
                           legacy=row.image_hash, legacy_method="legacy_rgb_tuple")
    color_root = Path(reference.path.iloc[0]).parent.parent
    for category in CLASSES:
        require((color_root / category).is_dir(), "Missing PV class folder")
        for path in sorted((color_root / category).rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                registry.photo(path, "plantvillage", path.name, category, FAMILIES[category],
                               "source_photo", str(color_root), "folder_enumeration")
    registry.coverage["pv_original_colour_files"] = sum(n["dataset"] == "plantvillage" for n in registry.nodes.values())
    # PP2020 originals and confirmed aliases, then overlay the actual C1 frozen roles.
    base = project / "data/field_manifests"
    pp_inventory, pp_path = registry.csv(base / "pp2020_audit/pp2020_inventory.csv", 1821)
    pp_lookup = pp_inventory.set_index("image_id")
    members, members_path = registry.csv(base / "pp2020_role_preservation/pp2020_all_file_role_registry.csv", 2533)
    registry.csv(base / "pp2020_role_preservation/pp2020_classification_role_registry.csv", 1821)
    registry.csv(base / "pp2020_role_preservation/protected_heldout_members.csv", 100)
    registry.csv(base / "pp2020_role_preservation/segmentation_training_classification_reconciliation.csv", 40)
    constraints, _ = registry.csv(base / "pp2020_duplicate_adjudication/pp2020_split_group_constraints.csv")
    adjudicated, _ = registry.csv(base / "pp2020_duplicate_adjudication/pp2020_duplicate_group_members.csv", 2533)
    require(set(adjudicated.image_path) == set(members.image_path), "PP2020 adjudication/role coverage differs")
    require(set(constraints.group_id) == set(members.group_id), "PP2020 group constraints differ")
    require(members.image_path.is_unique, "Duplicate PP2020 role paths")
    for i, row in enumerate(members.itertuples(index=False), 2):
        require(row.image_id in pp_lookup.index, "PP2020 alias lacks original labels")
        original = pp_lookup.loc[row.image_id]
        label = " ".join(flag for flag in ["healthy", "multiple_diseases", "rust", "scab"] if original[flag] == "1")
        role = {"segmentation_training": "segmenter_train", "evaluation_only": "protected_evaluation"}.get(row.member_role, "source_photo")
        registry.photo(row.image_path, "pp2020", row.image_id, label, label, role, members_path, i,
                       seed=f"pp2020:{row.group_id}", sticky=row.status.startswith("quarantine"))
    for row in pp_inventory.itertuples(index=False):
        node = registry.nodes[registry.by_path[str(Path(row.image_path).resolve())]]
        require(row.image_status == "ok", "PP2020 saved decoding error")
        node["expected_raw"].add(row.sha256)
        node["expected_legacy"].add(("legacy_exif_tuple", row.pixel_sha256))
        node["expected_dhash"].add(row.dhash256)
    prior, prior_path = registry.csv(base / "pp2020_audit/previous_use_registry.csv")
    for i, row in enumerate(prior.itertuples(index=False), 2):
        require(str(Path(row.image_path).resolve()) in registry.by_path, "Prior-use alias missing from PP2020 registry")
        node = registry.nodes[registry.by_path[str(Path(row.image_path).resolve())]]
        registry.photo(row.image_path, "pp2020", row.image_id, "", "", "historical_usage_registry", prior_path, i,
                       raw=row.sha256, legacy=row.pixel_sha256, legacy_method="legacy_exif_tuple", dhash=row.dhash256)
    report = json.loads(registry.source(base / "pp2020_split_v1/split_report.json").read_text())
    integrity = json.loads(registry.source(base / "pp2020_split_v1/split_integrity_report.json").read_text())
    require(integrity["passed"] and sha256(base / "pp2020_split_v1/split_report.json") == integrity["split_report_sha256"],
            "PP2020 frozen report changed")
    for entry in report["source_inputs"].values():
        require(sha256(registry.source(entry["path"])) == entry["sha256"], "PP2020 frozen provenance changed")
    for filename in ["train.csv", "val.csv", "test.csv", "excluded.csv"]:
        frame, path = registry.csv(base / "pp2020_split_v1" / filename)
        require(sha256(path) == report["manifest_sha256"][filename] == integrity["manifest_sha256"][filename], "PP2020 split changed")
        if filename == "excluded.csv":
            continue
        expected_count = {"train.csv": 1176, "val.csv": 251, "test.csv": 252}[filename]
        require(len(frame) == expected_count, "PP2020 frozen split count differs")
        role = {"train.csv": "classifier_train", "val.csv": "classifier_validation", "test.csv": "classifier_test"}[filename]
        for i, row in enumerate(frame.itertuples(index=False), 2):
            registry.photo(row.path, "pp2020", row.image_id, "", "", role, path, i,
                           raw=row.sha256, legacy=row.pixel_sha256, legacy_method="legacy_exif_tuple")
    # C1 manifests are actual completed training/model-selection evidence, never prospective permissions.
    mixed, mixed_path = registry.csv(c1 / "train_manifest.csv", 3390)
    require(mixed.path.is_unique, "Repeated C1 training path")
    require({str(Path(value).resolve()) for value in mixed.path} ==
            {n["path"] for n in registry.nodes.values() if "classifier_train" in n["roles"]},
            "C1 training omits/adds frozen source membership")
    for i, row in enumerate(mixed.itertuples(index=False), 2):
        require(str(Path(row.path).resolve()) in registry.by_path, "C1 training path outside registered source universe")
        require("classifier_train" in registry.nodes[registry.by_path[str(Path(row.path).resolve())]]["roles"],
                "C1 training membership contradicts frozen sources")
        require(CLASSES[int(row.label)] == row.class_name and FAMILIES[row.class_name] in
                registry.nodes[registry.by_path[str(Path(row.path).resolve())]]["label_families"], "C1 class mapping differs")
        registry.photo(row.path, row.source, Path(row.path).name, "", "", "classifier_train", mixed_path, i)
    for source, expected in [("plantvillage", 475), ("pp2020", 251)]:
        frame, path = registry.csv(c1 / f"{source}_validation_manifest.csv", expected)
        require(frame.path.is_unique, "Repeated C1 validation path")
        require({str(Path(value).resolve()) for value in frame.path} ==
                {n["path"] for n in registry.nodes.values() if n["dataset"] == source and "classifier_validation" in n["roles"]},
                "C1 validation omits/adds frozen source membership")
        for i, row in enumerate(frame.itertuples(index=False), 2):
            key = str(Path(row.path).resolve())
            require(key in registry.by_path and "classifier_validation" in registry.nodes[registry.by_path[key]]["roles"],
                    "C1 validation contradicts frozen membership")
            registry.photo(row.path, source, Path(row.path).name, "", "", "classifier_validation", path, i)
    # Lesion photographs and linked masks. Do not insert masks into the photograph index.
    for filename, expected in [("train.csv", 370), ("train_before_hard_negatives.csv", 330), ("val.csv", 15), ("test.csv", 30)]:
        frame, path = registry.csv(project / "lesion_manifests" / filename, expected)
        role = "segmenter_train" if filename.startswith("train") else "segmenter_validation" if filename.startswith("val") else "segmenter_test"
        for i, row in enumerate(frame.itertuples(index=False), 2):
            family = {"cedar_rust": "rust"}.get(row.disease, row.disease)
            node = registry.photo(row.image_path, "zenodo_synthetic" if row.source == "synthetic" else "zenodo_real",
                                  Path(row.image_path).name, row.disease, family, role, path, i)
            mask_path = (project / row.mask_path).resolve() if not Path(row.mask_path).is_absolute() else Path(row.mask_path).resolve()
            require(mask_path.is_file(), f"Missing lesion mask: {mask_path}")
            registry.auxiliary.append(dict(path=str(mask_path), kind="lesion_mask_not_photograph", linked_node=node["node_id"], source_file=path))
    for folder, dataset in [("restricted/Restricted_Plant_Village_Dataset/images", "zenodo_real"),
                            ("synthetic/Synthetic_Dataset/stable_diffusion_images", "zenodo_synthetic"),
                            ("synthetic/Synthetic_Dataset/blender_renders", "zenodo_render")]:
        root = project / "data/lesion_zenodo" / folder
        require(root.is_dir(), f"Missing Zenodo source: {root}")
        photos = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)
        require(len(photos) == (75 if dataset == "zenodo_real" else 300), "Zenodo image coverage differs")
        for path in photos:
            require(str(path.resolve()) in registry.by_path or dataset != "zenodo_real", "Unaccounted real lesion photograph")
            family = {"cedar_rust": "rust"}.get(path.parent.name, path.parent.name)
            registry.photo(path, dataset, path.relative_to(root).as_posix(), "" if dataset == "zenodo_real" else family,
                           "" if dataset == "zenodo_real" else family, "source_photo", str(root), "folder_enumeration")
    healthy, path = registry.csv(project / "data/healthy_structural_benchmark/manifest.csv", 50)
    for i, row in enumerate(healthy.itertuples(index=False), 2):
        node = registry.photo(row.image_path, "pp2020", Path(row.image_path).stem, "healthy", "healthy", "protected_evaluation", path, i)
        mask = (project / row.mask_path).resolve()
        require(mask.is_file(), "Healthy benchmark mask missing")
        registry.auxiliary.append(dict(path=str(mask), kind="healthy_mask_not_photograph", linked_node=node["node_id"], source_file=path))
    # Kashmir: preserve original readable aliases and recovered images, but never hash LFS text as an image.
    reviewed, reviewed_path = registry.csv(project / "results/2026-10-08_kashmir_label_audit/review_manifest.csv", 313)
    visual, _ = registry.csv(project / "artifacts/kashmir_label_audit/visual/manifest.csv", 313)
    require(reviewed[["review_id", "relative_path", "sha256"]].equals(visual[["review_id", "relative_path", "sha256"]]),
            "Kashmir review/inventory disagree")
    recovery = json.loads(registry.source(project / "artifacts/kashmir_label_audit/recovery/recovery.json").read_text())
    pointers = {row["relative_path"]: row for row in recovery["records"]}
    require(len(pointers) == 47, "Expected 47 recovered Kashmir objects")
    review_nodes = {}
    for i, row in enumerate(reviewed.itertuples(index=False), 2):
        paths = [project / "data/external_classification/kashmir_recovered" / row.relative_path]
        original = project / "data/external_classification/kashmir" / row.relative_path
        if row.relative_path in pointers:
            pointer = pointers[row.relative_path]
            require(sha256(original) == pointer["pointer_sha256"] and pointer["oid"] == row.sha256,
                    "Kashmir recovery/pointer identity changed")
            registry.auxiliary.append(dict(path=str(original.resolve()), kind="git_lfs_pointer_not_image",
                                           linked_node=row.review_id, source_file=reviewed_path))
        else:
            paths.append(original)
        for photo in paths:
            family = {"SCAB LEAVES": "scab", "HEALTHY LEAVES": "healthy"}.get(row.folder_label, "unverified_kashmir_rot")
            node = registry.photo(photo, "kashmir", row.review_id, row.folder_label, family, "external_evaluation", reviewed_path, i,
                                  raw=row.sha256, legacy=row.pixel_sha256, legacy_method="legacy_rgb_tuple",
                                  seed=f"kashmir:{row.conflict_group}" if row.conflict_group else "",
                                  sticky=row.draft_status == "exclude_conflicting_folder_labels", mapping_uncertain=True)
            review_nodes.setdefault(row.review_id, []).append(node["node_id"])
    pairs, pairs_path = registry.csv(project / "results/2026-10-08_kashmir_label_audit/cross_folder_pair_review.csv", 84)
    require(int(pairs.status.eq("visually_same_or_nearly_identical_photo_conflicting_folder_labels").sum()) == 12,
            "Kashmir adjudication coverage changed")
    for i, row in enumerate(pairs.itertuples(index=False), 2):
        if row.status == "visually_same_or_nearly_identical_photo_conflicting_folder_labels":
            for key in [row.a_id, row.b_id]:
                for node_id in review_nodes[key]:
                    registry.nodes[node_id]["seeds"].add(f"kashmir_pair:{row.pair_id}")
                    registry.evidence.append(dict(node_id=node_id, role="confirmed_prior_visual_conflict", source_file=pairs_path,
                                                  source_row=i, original_label="", seed_group=f"kashmir_pair:{row.pair_id}",
                                                  evidence_status="prior_recorded_visual_adjudication"))
    historical, path = registry.csv(project / "artifacts/kashmir_evaluation/efficientnet_b0/manifest.csv", 260)
    for i, row in enumerate(historical.itertuples(index=False), 2):
        require(str(Path(row.path).resolve()) in registry.by_path, "Historical Kashmir evaluation alias missing")
        registry.photo(row.path, "kashmir", row.relative_path, "", "", "external_evaluation", path, i, raw=row.sha256)
    # Source enumeration and registry are complete. Actual raw freshness verification occurs only in matching mode.
    require(len([n for n in registry.nodes.values() if n["dataset"] == "pp2021"]) == 18632, "PP2021 query universe incomplete")
    registry.coverage["reference_nodes_by_dataset"] = dict(Counter(n["dataset"] for n in registry.nodes.values() if n["dataset"] != "pp2021"))
    registry.coverage["auxiliary_references"] = len(registry.auxiliary)
    registry.coverage["source_metadata_files"] = len(registry.inputs)
    registry.coverage["c2b_counts"] = quarantine_summary
    return registry


def verify_raw(node):
    require(stat_signature(node["path"]) == node["stat"], f"Source changed before hashing: {node['path']}")
    raw = sha256(node["path"])
    require(not node["expected_raw"] or node["expected_raw"] == {raw}, f"Saved raw identity mismatch: {node['path']}")
    require(stat_signature(node["path"]) == node["stat"], f"Source changed during hashing: {node['path']}")
    return node["node_id"], raw


def hash_references(registry, workers):
    nodes = list(registry.nodes.values())
    print(f"Raw identity freshness: streaming {len(nodes)} files; PP2021 pixels/dHash will be reused", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (node_id, raw) in enumerate(pool.map(verify_raw, nodes), 1):
            registry.nodes[node_id]["raw_sha256"] = raw
            if index % 500 == 0 or index == len(nodes):
                print(f"Raw SHA verified {index}/{len(nodes)}", flush=True)
    refs = [node for node in nodes if node["dataset"] != "pp2021"]
    representatives = {}
    for node in refs:
        representatives.setdefault(node["raw_sha256"], node["path"])
    cache = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (raw, result) in enumerate(zip(representatives, pool.map(compatible_hashes, representatives.values())), 1):
            cache[raw] = result
            if index % 250 == 0 or index == len(representatives):
                print(f"Reference hash bridge {index}/{len(representatives)} distinct raw contents", flush=True)
    for node in refs:
        result = cache[node["raw_sha256"]]
        for method, expected in node["expected_legacy"]:
            require(result[method] == expected, f"Legacy pixel bridge failed: {node['path']}")
        require(not node["expected_dhash"] or node["expected_dhash"] == {result["dhash256"]},
                f"Saved dHash algorithm mismatch: {node['path']}")
        node.update(result)
    pp_raw = defaultdict(list)
    for node in nodes:
        if node["dataset"] == "pp2021":
            pp_raw[node["raw_sha256"]].append(node)
    parity_pairs = 0
    for raw in set(cache) & set(pp_raw):
        for node in pp_raw[raw]:
            require(all(str(cache[raw][key]) == str(node[key]) for key in ["pixel_sha256", "dhash256", "width", "height"]),
                    "Raw-identical PP2021/reference algorithms disagree")
            parity_pairs += 1
    report = dict(passed=True, pixel_method=PIXEL_METHOD, dhash_method=DHASH_METHOD, pillow_version=PILLOW_VERSION,
                  raw_identity_freshness="all registry photo bytes streamed and compared to saved digests where available",
                  pp2021_rows_reusing_saved_pixel_and_dhash=sum(n["dataset"] == "pp2021" for n in nodes), pp2021_wholesale_decoding_performed=False,
                  unique_reference_contents_decoded=len(cache), reference_photo_paths=len(refs),
                  raw_identical_pp2021_reference_parity_checks=parity_pairs,
                  legacy_header_hashes_directly_compared=False, incompatible_kashmir_phash_reused=False)
    return cache, report


def pair_record(left, right, distance=None):
    ids = sorted([left["node_id"], right["node_id"]])
    return dict(pair_id="PAIR_"+hashlib.sha256("\n".join(ids).encode()).hexdigest()[:24],
                left_id=left["node_id"], right_id=right["node_id"], left_dataset=left["dataset"], right_dataset=right["dataset"],
                left_original_id=left["original_id"], right_original_id=right["original_id"],
                left_path=left["path"], right_path=right["path"],
                left_labels=" | ".join(sorted(left["original_labels"])), right_labels=" | ".join(sorted(right["original_labels"])),
                left_roles=" | ".join(sorted(left["roles"])), right_roles=" | ".join(sorted(right["roles"])),
                left_raw_sha256=left["raw_sha256"], right_raw_sha256=right["raw_sha256"],
                left_pixel_sha256=left["pixel_sha256"], right_pixel_sha256=right["pixel_sha256"],
                left_dhash256=left["dhash256"], right_dhash256=right["dhash256"],
                raw_match=left["raw_sha256"] == right["raw_sha256"], pixel_match=left["pixel_sha256"] == right["pixel_sha256"],
                dhash_distance=(int(left["dhash256"], 16)^int(right["dhash256"], 16)).bit_count() if distance is None else distance,
                pixel_method=PIXEL_METHOD, dhash_method=DHASH_METHOD)


def match_nodes(nodes):
    """Exact matches + complete radius-four perceptual screen, including internal PP2021."""
    components = Components(nodes)
    seed_nodes, raw_nodes, pixel_nodes = defaultdict(list), defaultdict(list), defaultdict(list)
    refs, queries = [], []
    for node in nodes.values():
        for seed in node["seeds"]:
            seed_nodes[seed].append(node["node_id"])
        raw_nodes[node["raw_sha256"]].append(node["node_id"])
        pixel_nodes[node["pixel_sha256"]].append(node["node_id"])
        (queries if node["dataset"] == "pp2021" else refs).append(node)
    for collection in [seed_nodes, raw_nodes, pixel_nodes]:
        for members in collection.values():
            for member in members[1:]:
                components.join(members[0], member)
    exact, near, pending = {}, {}, set()
    # Inverted exact indices include every alias; emit PP2021/reference and unique PP2021/PP2021 pairs.
    for left in queries:
        hits = set(raw_nodes[left["raw_sha256"]]) | set(pixel_nodes[left["pixel_sha256"]])
        for right_id in sorted(hits):
            if right_id == left["node_id"] or (nodes[right_id]["dataset"] == "pp2021" and right_id < left["node_id"]):
                continue
            row = pair_record(left, nodes[right_id])
            row["status"] = "confirmed_exact_content_identity_labels_unresolved"
            exact[row["pair_id"]] = row
    ref_hashes, pp_hashes = defaultdict(list), defaultdict(list)
    for node in refs:
        ref_hashes[int(node["dhash256"], 16)].append(node)
    for node in queries:
        pp_hashes[int(node["dhash256"], 16)].append(node)
    ref_index, pp_index = HammingIndex(ref_hashes), HammingIndex(pp_hashes)
    for index, (value, left_nodes) in enumerate(sorted(pp_hashes.items()), 1):
        matches = [(other, distance, ref_hashes[other]) for other, distance in ref_index.query(value)]
        matches.extend((other, distance, pp_hashes[other]) for other, distance in pp_index.query(value) if other >= value)
        for _, distance, right_nodes in matches:
            for left in left_nodes:
                for right in right_nodes:
                    if left["node_id"] == right["node_id"]:
                        continue
                    if left["raw_sha256"] == right["raw_sha256"] or left["pixel_sha256"] == right["pixel_sha256"]:
                        continue
                    if left["dataset"] == right["dataset"] and value == int(right["dhash256"], 16) and right["node_id"] < left["node_id"]:
                        continue
                    # Existing reviewed or exact transitive component is already confirmed; retain provenance,
                    # but do not manufacture another unresolved hold for that established relationship.
                    row = pair_record(left, right, distance)
                    row["status"] = "previously_confirmed_component" if components.find(left["node_id"]) == components.find(right["node_id"]) else "pending_visual_review"
                    near[row["pair_id"]] = row
                    if row["status"] == "pending_visual_review":
                        pending.update([left["node_id"], right["node_id"]])
        if index % 2000 == 0 or index == len(pp_hashes):
            print(f"Indexed dHash screening {index}/{len(pp_hashes)} unique PP2021 hashes; {len(near)} nonexact pairs", flush=True)
    return (sorted(exact.values(), key=lambda row: row["pair_id"]), sorted(near.values(), key=lambda row: row["pair_id"]),
            components, pending, dict(reference_index=ref_index.summary(), internal_index=pp_index.summary(),
                                     unconditional_cross_product=len(queries)*len(refs), query_records=len(queries), pair_truncation=False))


def group_policy(members, pending):
    roles = set().union(*(node["roles"] for node in members))
    families = set().union(*(node["label_families"] for node in members))
    protected = bool(roles & {"classifier_validation", "classifier_test", "segmenter_validation", "segmenter_test", "protected_evaluation", "external_evaluation"})
    classifier_trained = "classifier_train" in roles
    segmenter_trained = "segmenter_train" in roles
    trained = classifier_trained or segmenter_trained
    review_pending = any(node["node_id"] in pending for node in members)
    mapping_pending = any(node["mapping_uncertain"] for node in members)
    # Frogeye and black_rot equivalence is provisional. Preserve a diagnostic hold rather than resolve it.
    incompatible_labels = len(families) > 1 and not families <= {"frog_eye_leaf_spot", "black_rot"}
    label_conflict = incompatible_labels or any(node["sticky_quarantine"] for node in members)
    role_conflict = trained and protected
    if label_conflict:
        status = "quarantined_label_conflict"
    elif role_conflict:
        status = "quarantined_prior_role_conflict"
    elif review_pending:
        status = "pending_visual_duplicate_review"
    elif protected:
        status = "protected_prior_validation_or_evaluation"
    elif trained:
        status = "previous_training_not_unseen"
    elif mapping_pending:
        status = "pending_diagnostic_mapping_review"
    else:
        status = "no_detected_overlap_pending_protocol_and_label_review"
    return dict(status=status, roles=" | ".join(sorted(roles)), label_families=" | ".join(sorted(families)),
                label_conflict=label_conflict, prior_role_conflict=role_conflict, protected_history=protected,
                classifier_training_seen=classifier_trained, segmenter_training_seen=segmenter_trained,
                review_pending=review_pending, diagnostic_mapping_pending=mapping_pending,
                forbidden_training_by_history=protected or label_conflict or role_conflict,
                forbidden_unseen_classifier_evaluation=classifier_trained or protected,
                forbidden_unseen_pipeline_evaluation=trained or protected,
                eligible_for_c2_training=False, eligible_for_unseen_evaluation=False,
                reason="intersection_of_all_confirmed_member_restrictions; pending_pairs_are_holds_not_duplicate_rejections")


def restrictions(nodes, components, pending):
    grouped = defaultdict(list)
    for node in nodes.values():
        grouped[components.find(node["node_id"])].append(node)
    memberships, policies, pp_rows = [], [], []
    for key, members in sorted(grouped.items()):
        ids = sorted(node["node_id"] for node in members)
        group_id = "CROSS_"+hashlib.sha256("\n".join(ids).encode()).hexdigest()[:24]
        policy = group_policy(members, pending)
        policies.append(dict(group_id=group_id, member_count=len(members), supporting_node_ids=" | ".join(ids), **policy))
        for node in members:
            memberships.append(dict(node_id=node["node_id"], group_id=group_id, dataset=node["dataset"],
                                    original_id=node["original_id"], path=node["path"],
                                    seed_groups=" | ".join(sorted(node["seeds"])), raw_sha256=node["raw_sha256"], pixel_sha256=node["pixel_sha256"]))
            if node["dataset"] == "pp2021":
                pp_rows.append(dict(image=node["original_id"], node_id=node["node_id"], group_id=group_id,
                                    original_labels=" | ".join(sorted(node["original_labels"])), proposed_label=node["proposed_label"],
                                    c2b_status=node["c2b_status"], reference_match_in_confirmed_group=any(n["dataset"] != "pp2021" for n in members),
                                    **policy))
    return pd.DataFrame(memberships), pd.DataFrame(policies), pd.DataFrame(pp_rows).sort_values("image")


def preservation_snapshot(project, excluded):
    """Protect original metadata, model bytes, and photo/auxiliary filesystem identities."""
    protected, stats = {}, {}
    files = set()
    for root in [project / name for name in ["data", "artifacts", "results", "lesion_manifests"]]:
        if root.is_dir():
            files.update(path for path in root.rglob("*") if path.is_file() and not any(
                ex == path.resolve() or ex in path.resolve().parents for ex in excluded))
    files.update(path for path in project.iterdir() if path.suffix.lower() in {".pt", ".pth", ".ckpt"})
    for path in sorted(files):
        if path.suffix.lower() in {".csv", ".json", ".pt", ".pth", ".ckpt"}:
            protected[str(path.resolve())] = sha256(path)
        elif path.suffix.lower() in EXTENSIONS | {".zip"}:
            stats[str(path.resolve())] = stat_signature(path)
    return dict(metadata_and_weights=protected, image_and_archive_stats=stats)


def save_csv(path, rows, columns=None):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, lineterminator="\n")


PAIR_COLUMNS = list(pair_record(dict(node_id="a", dataset="x", original_id="a", path="a", original_labels=set(), roles=set(),
                                     raw_sha256="a", pixel_sha256="a", dhash256="0"*64),
                                dict(node_id="b", dataset="y", original_id="b", path="b", original_labels=set(), roles=set(),
                                     raw_sha256="b", pixel_sha256="b", dhash256="0"*64))) + ["status"]


def review_sheets(output, pairs):
    pending = [row for row in pairs if row["status"] == "pending_visual_review"]
    folder = output / "review_sheets"
    folder.mkdir()
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 14) if font_path.is_file() else ImageFont.load_default()
    index_rows = []
    for start in range(0, len(pending), 6):
        page = Image.new("RGB", (1400, 6*230), "white")
        draw = ImageDraw.Draw(page)
        filename = f"pairs_{start//6+1:04d}.jpg"
        for offset, row in enumerate(pending[start:start+6]):
            y = offset*230
            draw.text((8, y+2), f"{row['pair_id']} | dHash {row['dhash_distance']}/256 | pending visual review", font=font, fill="black")
            for side, x in [("left", 8), ("right", 708)]:
                with Image.open(row[f"{side}_path"]) as raw:
                    raw.draft("RGB", (500, 300))
                    thumb = ImageOps.contain(ImageOps.exif_transpose(raw).convert("RGB"), (270, 190))
                page.paste(thumb, (x, y+28))
                text = f"{row[f'{side}_dataset']} / {row[f'{side}_original_id']}\n{row[f'{side}_labels']}\n{row[f'{side}_roles']}"
                # Wrap all provenance text rather than silently clipping long role strings.
                import textwrap
                text = "\n".join(textwrap.fill(line, width=48) for line in text.splitlines())
                draw.multiline_text((x+282, y+32), text, font=font, fill="black", spacing=4)
            index_rows.append(dict(pair_id=row["pair_id"], sheet=f"review_sheets/{filename}", tile=offset+1))
        page.save(folder / filename, quality=90)
        if (start//6+1) % 10 == 0:
            print(f"Review sheets {start//6+1}/{(len(pending)+5)//6}", flush=True)
    save_csv(output / "review_sheet_index.csv", index_rows, ["pair_id", "sheet", "tile"])
    return len(index_rows)


def verify_existing(output, registry, script_hash):
    report = json.loads((output / "cross_dataset_integrity_report.json").read_text())
    require(report["complete"] and report["script_sha256"] == script_hash, "Existing audit incomplete/different implementation; use a new version")
    require(report["source_inputs"] == registry.inputs, "Existing audit inputs differ; use a new version")
    actual = {path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob("*"))
              if path.is_file() and path.name != "cross_dataset_integrity_report.json"}
    require(actual == report["output_sha256"], "Existing audit output modified; preserve it and use a new version")
    receipts = pd.read_csv(output / "photo_freshness_receipts.csv", dtype=str, keep_default_na=False)
    require(set(receipts.node_id) == set(registry.nodes), "Existing photo coverage differs")
    for row in receipts.itertuples(index=False):
        node = registry.nodes[row.node_id]
        require(row.path == node["path"] and [int(row.size), int(row.mtime_ns), int(row.ctime_ns)] == node["stat"],
                f"Photo changed since audit: {row.path}; use a new version")
    print("PASS: existing complete audit verified and reused; no files rewritten", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--quarantine-dir", type=Path, default=Path("data/field_manifests/pp2021_audit/internal_quarantine_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/field_manifests/pp2021_cross_dataset_v1"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    require(args.workers > 0, "Positive workers required")
    project = args.project.resolve()
    output = (args.output_dir if args.output_dir.is_absolute() else project / args.output_dir).resolve()
    quarantine_dir = (args.quarantine_dir if args.quarantine_dir.is_absolute() else project / args.quarantine_dir).resolve()
    require(output.parent == project / "data/field_manifests" and bool(re.fullmatch(r"pp2021_cross_dataset_v[1-9]\d*", output.name)),
            "Use a versioned output under data/field_manifests/pp2021_cross_dataset_vN")
    require(not output.is_symlink(), "Output symlink forbidden")
    script_hash = sha256(__file__)
    print("Validating quarantine, source coverage, frozen roles and input schemas...", flush=True)
    registry = build_registry(project, quarantine_dir)
    print(json.dumps(registry.coverage, indent=2), flush=True)
    require(all(len(n["expected_raw"]) <= 1 for n in registry.nodes.values()), "Inconsistent expected raw identities")
    if args.validate_only:
        print("VALIDATION PASS: source roles/coverage and C2-B provenance; raw-byte freshness/bridge verified in matching mode. No output writes.", flush=True)
        return
    if output.exists():
        verify_existing(output, registry, script_hash)
        return
    stage = output.with_name("."+output.name+"_building")
    require(not stage.exists(), f"Existing incomplete staging directory preserved: {stage}; choose a new version")
    print("Recording preservation snapshot (metadata, weights and image file metadata)...", flush=True)
    before = preservation_snapshot(project, [output, stage])
    stage.mkdir()
    # On failure keep this explicit incomplete staging directory for inspection; never publish partial results.
    try:
        cache, compatibility = hash_references(registry, args.workers)
        save_csv(stage / "photo_freshness_receipts.csv", [dict(node_id=n["node_id"], path=n["path"], raw_sha256=n["raw_sha256"],
                 size=n["stat"][0], mtime_ns=n["stat"][1], ctime_ns=n["stat"][2]) for n in registry.nodes.values()])
        save_csv(stage / "reference_hash_cache.csv", [dict(raw_sha256=raw, **result) for raw, result in sorted(cache.items())])
        save_csv(stage / "reference_registry.csv", registry.table())
        save_csv(stage / "role_evidence.csv", registry.evidence)
        save_csv(stage / "auxiliary_sources.csv", registry.auxiliary)
        (stage / "hash_compatibility_report.json").write_bytes(json_bytes(compatibility))
        exact, near, components, pending, index_report = match_nodes(registry.nodes)
        save_csv(stage / "exact_matches.csv", exact, PAIR_COLUMNS)
        save_csv(stage / "perceptual_candidates.csv", near, PAIR_COLUMNS)
        template = [dict(pair_id=row["pair_id"], left_raw_sha256=row["left_raw_sha256"], right_raw_sha256=row["right_raw_sha256"],
                         reviewer="", decision="unresolved", reason="", evidence="", review_date="")
                    for row in near if row["status"] == "pending_visual_review"]
        save_csv(stage / "review_decisions_template.csv", template,
                 ["pair_id", "left_raw_sha256", "right_raw_sha256", "reviewer", "decision", "reason", "evidence", "review_date"])
        memberships, policies, status = restrictions(registry.nodes, components, pending)
        require(len(status) == 18632 and status.image.is_unique, "Overlap status lost PP2021 records")
        require(int(status.c2b_status.eq("quarantined_label_conflict").sum()) == 54, "C2-B quarantine lost")
        require(status[status.c2b_status.eq("quarantined_label_conflict")].status.eq("quarantined_label_conflict").all(),
                "C2-B conflict was improperly cleared")
        require(not status.eligible_for_c2_training.any(), "Audit cannot approve training")
        save_csv(stage / "confirmed_group_members.csv", memberships)
        save_csv(stage / "group_restrictions.csv", policies)
        save_csv(stage / "pp2021_overlap_status.csv", status)
        sheets = review_sheets(stage, near)
        pending_candidates = status[status.c2b_status.eq("pending_cross_dataset_audit")]
        summary = dict(pp2021_records=18632, c2b_quarantined_records=54, original_candidates=14491,
                       c2b_removed_candidates=27, c2b_remaining_candidates=len(pending_candidates),
                       exact_pair_count=len(exact), exact_cross_dataset_pairs=sum(r["right_dataset"] != "pp2021" for r in exact),
                       exact_internal_pp2021_pairs=sum(r["right_dataset"] == "pp2021" for r in exact),
                       nonexact_perceptual_pairs=len(near), pairs_pending_visual_review=len(template), review_sheet_tiles=sheets,
                       pp2021_confirmed_reference_overlap_records=int(status.reference_match_in_confirmed_group.sum()),
                       pp2021_status_counts=status.status.value_counts().to_dict(),
                       remaining_candidate_status_counts=pending_candidates.status.value_counts().to_dict(),
                       remaining_candidate_source_label_counts=pending_candidates.original_labels.value_counts().to_dict(),
                       approved_for_training=0, assigned_splits=0, training_performed=False,
                       audit_status="matching_complete_visual_and_diagnostic_review_pending",
                       limitations=["dHash <=4 is a review screen, not automatic rejection or proof of independence",
                                    "Frogeye/black-rot diagnostic equivalence remains provisional",
                                    "Historical dataset use is bounded by supplied local records"])
        (stage / "eligibility_summary.json").write_bytes(json_bytes(summary))
        print("Checking preservation and final output integrity...", flush=True)
        for node in registry.nodes.values():
            require(stat_signature(node["path"]) == node["stat"], f"Photo changed during matching: {node['path']}")
        require(before == preservation_snapshot(project, [output, stage]), "Protected files changed; audit not published")
        require(all(sha256(path) == expected for path, expected in registry.inputs.items()), "Source metadata changed; audit not published")
        require(sha256(__file__) == script_hash, "Implementation changed during matching; audit not published")
        report = dict(complete=True, integrity_passed=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
                      script_sha256=script_hash, source_inputs=registry.inputs, coverage=registry.coverage,
                      hash_compatibility=compatibility, indexing=index_report,
                      original_metadata_and_weights_unchanged=True, original_photo_file_metadata_unchanged=True,
                      protected_metadata_and_weight_files=len(before["metadata_and_weights"]),
                      protected_image_archive_paths=len(before["image_and_archive_stats"]),
                      metadata_weight_fingerprint=hashlib.sha256(json_bytes(before["metadata_and_weights"])).hexdigest(),
                      image_stat_fingerprint=hashlib.sha256(json_bytes(before["image_and_archive_stats"])).hexdigest(),
                      training_performed=False, splits_assigned=False, new_visual_decisions_applied=0,
                      output_sha256={p.relative_to(stage).as_posix(): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()})
        (stage / "cross_dataset_integrity_report.json").write_bytes(json_bytes(report))
        require(stage.parent == output.parent == project / "data/field_manifests" and not output.exists(), "Unsafe publication target")
        stage.rename(output)
        print(json.dumps(summary, indent=2), flush=True)
        print(f"COMPLETE: {output}; no training/splits/automatic visual adjudication", flush=True)
    except Exception as error:
        (stage / "INCOMPLETE.json").write_bytes(json_bytes(dict(complete=False, error=f"{type(error).__name__}: {error}")))
        raise


if __name__ == "__main__":
    main()
