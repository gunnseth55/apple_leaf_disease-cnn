"""C2-C exact identities, indexed screening, source preservation and role propagation."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, PngImagePlugin


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("pp2021_cross_dataset", SCRIPTS / "audit_pp2021_cross_dataset.py")
audit = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC.loader.exec_module(audit)
finally:
    sys.path.pop(0)


def node(key, dataset="pp2021", raw=None, pixel=None, dhash=0, role="pp2021_candidate", family="scab", seed="", sticky=False):
    return dict(node_id=key, dataset=dataset, original_id=key, path=key+".png", original_labels={family}, label_families={family},
                roles={role}, seeds={seed} if seed else set(), raw_sha256=raw or key, pixel_sha256=pixel or key,
                dhash256=f"{dhash:064x}", sticky_quarantine=sticky, mapping_uncertain=False, proposed_label="Apple___Apple_scab",
                c2b_status="pending_cross_dataset_audit" if role == "pp2021_candidate" else "excluded_original_label")


class PP2021CrossDatasetTests(unittest.TestCase):
    def test_hash_bridge_same_pixels_different_metadata_and_headers(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary)/"a.png", Path(temporary)/"renamed.png"
            image = Image.fromarray(np.random.default_rng(11).integers(0, 256, (33, 47, 3), dtype=np.uint8))
            image.save(first)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("note", "different file bytes")
            image.save(second, pnginfo=metadata)
            self.assertNotEqual(audit.sha256(first), audit.sha256(second))
            left, right = audit.compatible_hashes(first), audit.compatible_hashes(second)
            self.assertEqual(left["pixel_sha256"], right["pixel_sha256"])
            self.assertEqual(left["dhash256"], right["dhash256"])
            self.assertNotEqual(left["pixel_sha256"], left["legacy_rgb_tuple"])

    def test_exif_normalization_and_manual_dhash_bit_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, corrected = Path(temporary)/"rotated.jpg", Path(temporary)/"corrected.png"
            image = Image.fromarray(np.random.default_rng(9).integers(0, 256, (60, 80, 3), dtype=np.uint8))
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            before = path.read_bytes()
            with Image.open(path) as original:
                rgb = ImageOps.exif_transpose(original).convert("RGB")
                rgb.save(corrected)
                gray = np.asarray(rgb.convert("L").resize((17, 16), Image.Resampling.LANCZOS))
            result = audit.compatible_hashes(path)
            self.assertEqual((result["width"], result["height"]), (60, 80))
            self.assertNotEqual(result["legacy_rgb_tuple"], result["legacy_exif_tuple"])
            self.assertEqual(result["pixel_sha256"], audit.compatible_hashes(corrected)["pixel_sha256"])
            manual = 0
            for value in (gray[:, 1:] > gray[:, :-1]).flatten():
                manual = (manual << 1) | int(value)
            self.assertEqual(result["dhash256"], f"{manual:064x}")
            self.assertEqual(before, path.read_bytes())

    def test_index_complete_against_brute_force_including_boundary_four(self):
        rng = np.random.default_rng(47)
        base = [int.from_bytes(rng.bytes(32), "big") for _ in range(70)]
        # Each four-bit mutation affects four distinct blocks, leaving exactly one exact bucket.
        mask4 = sum(1 << shift for shift in [1, 53, 104, 155])
        values = set(base + [value ^ mask4 for value in base] + [0, 15, 31])
        index = audit.HammingIndex(values)
        for value in base + [0, 15, 31]:
            self.assertEqual(index.query(value), sorted((other, (value^other).bit_count()) for other in values if (value^other).bit_count() <= 4))
        self.assertIn((15, 4), index.query(0))
        self.assertNotIn((31, 5), index.query(0))
        self.assertFalse(index.summary()["truncated"])

    def test_exact_raw_and_pixel_matches_expand_all_alias_roles(self):
        nodes = {n["node_id"]: n for n in [node("pp", raw="same", pixel="same"),
                 node("train", "pp2020", raw="same", pixel="same", role="classifier_train"),
                 node("test", "pp2020", raw="otherbytes", pixel="same", role="classifier_test")]}
        exact, near, components, pending, _ = audit.match_nodes(nodes)
        self.assertEqual(len(exact), 2)
        self.assertEqual(sum(row["raw_match"] for row in exact), 1)
        self.assertFalse(near)
        _, policies, status = audit.restrictions(nodes, components, pending)
        row = status.iloc[0]
        self.assertTrue(row.prior_role_conflict)
        self.assertTrue(row.protected_history)
        self.assertTrue(row.forbidden_training_by_history)
        self.assertTrue(row.forbidden_unseen_classifier_evaluation)
        self.assertFalse(row.eligible_for_c2_training)

    def test_excluded_pp2021_bridge_propagates_protected_history_and_conflict(self):
        nodes = {n["node_id"]: n for n in [node("candidate", pixel="identity", raw="a"),
                 node("excluded", pixel="identity", raw="b", role="excluded_original_label", family="scab complex"),
                 node("protected", "pp2020", pixel="identity", raw="c", role="classifier_validation")]}
        exact, _, components, pending, _ = audit.match_nodes(nodes)
        _, _, status = audit.restrictions(nodes, components, pending)
        self.assertEqual(len(exact), 3)
        self.assertTrue(status.protected_history.all())
        self.assertTrue(status.status.eq("quarantined_label_conflict").all())

    def test_perceptual_hits_do_not_union_or_automatically_reject(self):
        nodes = {n["node_id"]: n for n in [node("candidate", dhash=0),
                 node("protected", "pp2020", dhash=15, role="classifier_test")]}
        exact, near, components, pending, _ = audit.match_nodes(nodes)
        self.assertEqual(len(near), 1)
        self.assertFalse(exact)
        self.assertNotEqual(components.find("candidate"), components.find("protected"))
        _, _, status = audit.restrictions(nodes, components, pending)
        row = status.iloc[0]
        self.assertEqual(row.status, "pending_visual_duplicate_review")
        self.assertFalse(row.protected_history)
        self.assertFalse(row.label_conflict)
        self.assertFalse(row.eligible_for_c2_training)

    def test_internal_screen_is_complete_without_self_or_reverse_pairs(self):
        nodes = {key: node(key, dhash=value) for key, value in [("a", 0), ("b", 15), ("c", 31)]}
        _, near, _, _, report = audit.match_nodes(nodes)
        pairs = {frozenset([row["left_id"], row["right_id"]]) for row in near}
        self.assertEqual(pairs, {frozenset(["a", "b"]), frozenset(["b", "c"])})
        self.assertEqual(len(near), len(pairs))
        self.assertEqual(report["query_records"], 3)

    def test_sticky_quarantine_and_provisional_diagnosis_not_cleared(self):
        sticky = node("sticky", sticky=True)
        self.assertEqual(audit.group_policy([sticky], set())["status"], "quarantined_label_conflict")
        frogeye, blackrot = node("frog", family="frog_eye_leaf_spot"), node("black", "plantvillage", family="black_rot")
        frogeye["mapping_uncertain"] = True
        result = audit.group_policy([frogeye, blackrot], set())
        self.assertTrue(result["diagnostic_mapping_pending"])
        self.assertFalse(result["eligible_for_c2_training"])
        self.assertFalse(result["label_conflict"])
        segmenter = node("segmenter", "zenodo_real", role="segmenter_train")
        result = audit.group_policy([segmenter], set())
        self.assertFalse(result["forbidden_unseen_classifier_evaluation"])
        self.assertTrue(result["forbidden_unseen_pipeline_evaluation"])

    def test_changed_source_fails_raw_freshness(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"image.png"
            Image.new("RGB", (8, 8), "green").save(path)
            value = dict(path=str(path), stat=audit.stat_signature(path), expected_raw={"f"*64}, node_id="a")
            with self.assertRaisesRegex(ValueError, "Saved raw identity mismatch"):
                audit.verify_raw(value)
            value["expected_raw"] = {audit.sha256(path)}
            self.assertEqual(audit.verify_raw(value)[1], audit.sha256(path))
            Image.new("RGB", (12, 10), "blue").save(path)
            with self.assertRaisesRegex(ValueError, "changed before hashing"):
                audit.verify_raw(value)

    def test_registry_preserves_roles_rejects_incompatible_hash_and_missing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/"image.png"
            Image.new("RGB", (8, 8), "green").save(path)
            registry = audit.Registry(root)
            registry.photo(path, "pp2020", "a", "scab", "scab", "classifier_train", "fixture", 1)
            registry.photo(path, "pp2020", "a", "scab", "scab", "classifier_validation", "fixture", 2)
            self.assertEqual(len(registry.nodes), 1)
            self.assertEqual(next(iter(registry.nodes.values()))["roles"], {"classifier_train", "classifier_validation"})
            with self.assertRaisesRegex(ValueError, "dHash256"):
                registry.photo(path, "pp2020", "a", "", "", "source_photo", "fixture", 3, dhash="0"*16)
            with self.assertRaisesRegex(ValueError, "Required source missing"):
                registry.csv("missing.csv")

    def test_raw_alias_cache_and_all_source_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root/"a.png", root/"b.png"
            Image.new("RGB", (8, 8), "green").save(first)
            second.write_bytes(first.read_bytes())
            registry = audit.Registry(root)
            registry.photo(first, "reference", "a", "scab", "scab", "source_photo", "fixture", 1)
            registry.photo(second, "reference", "b", "scab", "scab", "classifier_test", "fixture", 2)
            before = {p: p.read_bytes() for p in [first, second]}
            with redirect_stdout(io.StringIO()):
                cache, report = audit.hash_references(registry, 2)
            self.assertEqual(len(cache), 1)
            self.assertEqual(report["unique_reference_contents_decoded"], 1)
            self.assertEqual(report["pp2021_rows_reusing_saved_pixel_and_dhash"], 0)
            self.assertEqual(before, {p: p.read_bytes() for p in [first, second]})

    def test_safe_output_and_existing_incomplete_version_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "versioned output"):
                audit.main(["--project", temporary, "--output-dir", str(Path(temporary)/"train")])

    def test_existing_audit_checks_outputs_inputs_and_photo_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo = root/"image.png"
            Image.new("RGB", (8, 8), "green").save(photo)
            registry = audit.Registry(root)
            value = registry.photo(photo, "reference", "a", "scab", "scab", "source_photo", "fixture", 1)
            output = root/"audit"
            output.mkdir()
            receipt = output/"photo_freshness_receipts.csv"
            audit.save_csv(receipt, [dict(node_id=value["node_id"], path=value["path"], raw_sha256=audit.sha256(photo),
                                         size=value["stat"][0], mtime_ns=value["stat"][1], ctime_ns=value["stat"][2])])
            report = dict(complete=True, script_sha256="source", source_inputs={}, output_sha256={receipt.name: audit.sha256(receipt)})
            report_path = output/"cross_dataset_integrity_report.json"
            report_path.write_bytes(audit.json_bytes(report))
            before = {path: path.read_bytes() for path in output.iterdir()}
            with redirect_stdout(io.StringIO()):
                audit.verify_existing(output, registry, "source")
            self.assertEqual(before, {path: path.read_bytes() for path in output.iterdir()})
            with self.assertRaisesRegex(ValueError, "different implementation"):
                audit.verify_existing(output, registry, "different")
            value["stat"][0] += 1
            with self.assertRaisesRegex(ValueError, "Photo changed"):
                audit.verify_existing(output, registry, "source")
            value["stat"][0] -= 1
            receipt.write_bytes(receipt.read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError, "output modified"):
                audit.verify_existing(output, registry, "source")

    def test_preservation_snapshot_detects_checkpoint_split_or_auxiliary_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root/"artifacts/frozen.pt"
            split = root/"data/field_manifests/frozen/test.csv"
            mask = root/"data/mask.png"
            for path in [checkpoint, split, mask]:
                path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"frozen weights")
            split.write_bytes(b"image,role\na,test\n")
            Image.new("L", (8, 8)).save(mask)
            baseline = audit.preservation_snapshot(root, [])
            for path in [checkpoint, split, mask]:
                before = path.read_bytes()
                path.write_bytes(before+b"changed")
                self.assertNotEqual(baseline, audit.preservation_snapshot(root, []))
                path.write_bytes(before)
                baseline = audit.preservation_snapshot(root, [])


if __name__ == "__main__":
    unittest.main()
