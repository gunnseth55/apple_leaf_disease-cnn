"""C2-B quarantine, provenance, preservation and fail-closed publication checks."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pp2021_quarantine", PROJECT / "scripts/quarantine_pp2021_conflicts.py")
quarantine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quarantine)


class PP2021QuarantineTests(unittest.TestCase):
    def fixture(self):
        categories = ["rust", "rust complex", "scab", "frog_eye_leaf_spot", "healthy", "complex"]
        labels = pd.DataFrame([dict(image=f"{i:03d}.jpg", labels=label) for i, label in enumerate(categories)])
        candidates = labels[labels.labels.isin(quarantine.MAPPING)].copy()
        candidates["proposed_label"] = candidates.labels.map(quarantine.MAPPING)
        candidates["review_status"] = "pending_audit"
        inventory = pd.DataFrame([dict(image=row.image, original_labels=row.labels,
                                      proposed_label=quarantine.MAPPING.get(row.labels, ""),
                                      status="ok" if row.labels in quarantine.MAPPING else "excluded_label",
                                      width="8", height="8", file_sha256=f"{i+100:064x}",
                                      pixel_sha256=f"{0 if i < 2 else i:064x}", dhash256=f"{i:064x}", error="")
                                  for i, row in enumerate(labels.itertuples())])
        members = inventory.iloc[:2].copy()
        members["group_id"] = "PP2021_DUP_001"
        members["label_conflict"] = "True"
        groups = pd.DataFrame([dict(group_id="PP2021_DUP_001", image_count="2", image_ids="000.jpg | 001.jpg",
                                    original_labels="rust | rust complex", label_conflict="True", candidate_count="1",
                                    excluded_count="1", status="pending_review")])
        expected = dict(original_records=6, original_candidates=4, unique_pixels=5, conflicting_groups=1,
                        quarantined_records=2, quarantined_candidates=1, remaining_candidates=3)
        return dict(labels=labels, candidates=candidates, inventory=inventory, groups=groups, members=members), expected

    def project_fixture(self, root):
        tables, expected = self.fixture()
        for key, relative in quarantine.SOURCES.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            tables[key].to_csv(path, index=False)
        image_dir = root / "data/plant_pathology_2021/train_images"
        image_dir.mkdir()
        for image in tables["labels"].image:
            (image_dir / image).write_bytes(b"source bytes must not be read or rewritten")
        checkpoint = root / "artifacts/frozen.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"unchanged checkpoint")
        split = root / "data/field_manifests/pp2020_split_v1/test.csv"
        split.parent.mkdir()
        split.write_text("image,role\nprotected,test\n")
        legacy = root / "data/field_manifests/pp2021_audit/candidates_after_internal_audit.csv"
        legacy.write_text("existing,partial\nkept,unchanged\n")
        return expected

    def quiet_run(self, root, expected, output=None):
        with redirect_stdout(io.StringIO()):
            return quarantine.run(root, output, expected)

    def test_counts_partition_labels_provenance_and_pending_permissions(self):
        tables, expected = self.fixture()
        outputs, report = quarantine.build_outputs(tables, expected)
        all_removed, removed, remaining = [outputs[name] for name in quarantine.OUTPUTS[:3]]
        self.assertEqual([len(all_removed), len(removed), len(remaining)], [2, 1, 3])
        self.assertEqual(set(all_removed.image), {"000.jpg", "001.jpg"})
        self.assertFalse(set(all_removed.image) & set(remaining.image))
        original = tables["labels"].set_index("image").labels
        for frame in outputs.values():
            self.assertTrue(frame.image.is_unique)
            self.assertTrue(frame.labels.eq(frame.image.map(original)).all())
            self.assertFalse(frame.eligible_for_training.any())
            self.assertFalse(frame.eligible_for_unseen_evaluation.any())
        self.assertTrue(all_removed.group_id.ne("").all())
        self.assertTrue(all_removed.quarantine_reason.eq(quarantine.REASON).all())
        self.assertTrue(all_removed.status.eq(quarantine.QUARANTINE).all())
        self.assertTrue(remaining.review_status.eq(quarantine.PENDING).all())
        self.assertEqual(report["remaining_candidate_class_counts"], {"frog_eye_leaf_spot": 1, "healthy": 1, "scab": 1})
        frogeye = remaining[remaining.labels.eq("frog_eye_leaf_spot")].iloc[0]
        self.assertEqual(frogeye.proposed_label, "Apple___Black_rot")
        self.assertIn("provisional", frogeye.mapping_status)
        self.assertEqual(all_removed.loc[all_removed.image.eq("001.jpg"), "proposed_label"].iloc[0], "")

    def test_wrong_expected_totals_fail(self):
        tables, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "Unexpected original record count"):
            quarantine.build_outputs(tables)

    def test_source_or_proposed_label_mutations_fail(self):
        for field, replacement, message in [("labels", "healthy", "original labels"),
                                             ("proposed_label", "Apple___healthy", "proposed labels")]:
            tables, expected = self.fixture()
            tables["candidates"].loc[0, field] = replacement
            with self.assertRaisesRegex(ValueError, message):
                quarantine.build_outputs(tables, expected)

    def test_group_or_member_tampering_fails(self):
        for table, column, replacement, message in [
            ("groups", "label_conflict", "maybe", "boolean"),
            ("groups", "image_ids", "000.jpg | 002.jpg", "IDs disagree"),
            ("members", "pixel_sha256", "f"*64, "inventory audit fields"),
            ("groups", "candidate_count", "2", "counts disagree"),
        ]:
            tables, expected = self.fixture()
            tables[table].iloc[0, tables[table].columns.get_loc(column)] = replacement
            with self.assertRaisesRegex(ValueError, message):
                quarantine.build_outputs(tables, expected)

    def test_missing_members_duplicate_ids_and_images_fail(self):
        tables, expected = self.fixture()
        tables["members"] = tables["members"].iloc[:1]
        with self.assertRaisesRegex(ValueError, "membership/count"):
            quarantine.build_outputs(tables, expected)
        tables, expected = self.fixture()
        tables["candidates"] = pd.concat([tables["candidates"].iloc[:3], tables["candidates"].iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate/empty image IDs"):
            quarantine.build_outputs(tables, expected)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.project_fixture(root)
            (root / "data/plant_pathology_2021/train_images/000.jpg").unlink()
            with self.assertRaisesRegex(ValueError, "source image is missing"):
                self.quiet_run(root, expected)
            self.assertFalse((root / "data/field_manifests/pp2021_audit/internal_quarantine_v1").exists())

    def test_publication_reuse_and_sources_checkpoints_frozen_splits_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.project_fixture(root)
            before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.quiet_run(root, expected)
            output = root / "data/field_manifests/pp2021_audit/internal_quarantine_v1"
            times = {path.name: path.stat().st_mtime_ns for path in output.iterdir()}
            self.quiet_run(root, expected)
            self.assertEqual(times, {path.name: path.stat().st_mtime_ns for path in output.iterdir()})
            for path, data in before.items():
                self.assertEqual(Path(path).read_bytes(), data)
            changed = output / quarantine.OUTPUTS[0]
            changed.write_bytes(changed.read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError, "Existing output differs"):
                self.quiet_run(root, expected)
            self.assertTrue(changed.read_bytes().endswith(b"tamper"))
            # A deliberately new version can be published without touching the earlier one.
            self.quiet_run(root, expected, output.with_name("internal_quarantine_v2"))
            self.assertTrue(changed.read_bytes().endswith(b"tamper"))

    def test_protected_change_stops_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.project_fixture(root)
            original = quarantine.build_outputs
            def mutate(*args):
                result = original(*args)
                (root / "artifacts/frozen.pt").write_bytes(b"external mutation")
                return result
            with patch.object(quarantine, "build_outputs", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "Protected inputs changed"):
                    self.quiet_run(root, expected)
            self.assertFalse((root / "data/field_manifests/pp2021_audit/internal_quarantine_v1").exists())

    def test_output_cannot_escape_versioned_audit_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.project_fixture(root)
            with self.assertRaisesRegex(ValueError, "explicitly versioned"):
                self.quiet_run(root, expected, root / "data/plant_pathology_2021")

    def test_actual_pp2021_all_ten_requested_integrity_checks(self):
        if not all((PROJECT / path).is_file() for path in quarantine.SOURCES.values()):
            self.skipTest("Local PP2021 audit inputs are unavailable")
        before = {key: quarantine.sha256(PROJECT / path) for key, path in quarantine.SOURCES.items()}
        tables = {key: pd.read_csv(PROJECT / path, dtype=str, keep_default_na=False) for key, path in quarantine.SOURCES.items()}
        outputs, report = quarantine.build_outputs(tables)
        self.assertEqual(report["conflicting_groups"], 27)
        self.assertEqual([len(outputs[name]) for name in quarantine.OUTPUTS[:3]], [54, 27, 14464])
        self.assertEqual(report["quarantined_candidate_class_counts"], {"rust": 27})
        self.assertEqual(report["remaining_candidate_class_counts"],
                         {"scab": 4826, "frog_eye_leaf_spot": 3181, "rust": 1833, "healthy": 4624})
        self.assertEqual(before, {key: quarantine.sha256(PROJECT / path) for key, path in quarantine.SOURCES.items()})


if __name__ == "__main__":
    unittest.main()
