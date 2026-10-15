"""Audit failure modes and contamination safeguards on tiny local fixtures."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image, PngImagePlugin

SPEC = importlib.util.spec_from_file_location("pp2020_audit", Path(__file__).resolve().parents[1] / "scripts/audit_pp2020_for_field_training.py")
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


class PP2020AuditTests(unittest.TestCase):
    def fixture(self, project):
        images = project / "data/plant_pathology_2020/extracted/images"
        images.mkdir(parents=True)
        categories = ["healthy", "healthy", "rust", "scab", "healthy", "scab", "multiple_diseases", "healthy", "healthy"]
        rows = []
        for index, category in enumerate(categories):
            rng = np.random.default_rng(index + 12)
            Image.fromarray(rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)).save(images / f"Train_{index}.png")
            rows.append({"image_id": f"Train_{index}", **{flag: int(category == flag) for flag in audit_module.FLAGS}})
        # Same decoded image with different metadata / bytes, renamed to another ID.
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("note", "renamed evaluation copy")
        with Image.open(images / "Train_1.png") as source:
            source.save(images / "Train_4.png", pnginfo=metadata)
            changed = np.array(source)
        changed[0, 0] = (changed[0, 0].astype(int) + 1) % 256
        Image.fromarray(changed).save(images / "Train_7.png")
        # Corrupt bytes and invalid labels must both remain recorded.
        (images / "Train_5.png").write_bytes(b"not an image")
        rows[5]["healthy"] = 1
        rows[8]["healthy"] = "missing"
        pd.DataFrame(rows).to_csv(images.parent / "train.csv", index=False)
        # Deliberately unlabelled challenge test metadata must not be used.
        pd.DataFrame([{"image_id": "Test_0"}]).to_csv(images.parent / "test.csv", index=False)
        for purpose, index, source, manifest in [
            ("training", 0, "hard_negative", "lesion_manifests/train.csv"),
            ("evaluation", 1, "evaluation_only", "data/healthy_structural_benchmark/manifest.csv"),
        ]:
            destination = project / "data" / purpose / f"Train_{index}.png"
            destination.parent.mkdir(parents=True)
            shutil.copyfile(images / f"Train_{index}.png", destination)
            csv = project / manifest
            csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"image_path": str(destination), "source": source}]).to_csv(csv, index=False)
        cedar = project / "data/external_classification/cedar_pp2020"
        cedar.mkdir(parents=True)
        shutil.copyfile(images / "Train_2.png", cedar / "Train_2.png")
        return images

    def run_audit(self, project):
        with redirect_stdout(io.StringIO()):
            return audit_module.audit(project, project / "audit", expected_training=1, expected_eval=1)

    def test_registry_duplicates_integrity_and_no_source_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            images = self.fixture(project)
            before = {str(p): audit_module.sha256(p) for p in images.iterdir()}
            frame, summary = self.run_audit(project)
            indexed = frame.set_index("image_id")
            self.assertEqual(len(frame), 9)
            self.assertFalse(summary["challenge_test_used"])
            self.assertFalse(summary["splits_assigned"])
            self.assertTrue(indexed.loc["Train_0", "eligible_training_candidate"])
            self.assertFalse(indexed.loc["Train_0", "eligible_unseen_evaluation"])
            self.assertTrue(indexed.loc["Train_3", "eligible_training_candidate"])
            for image_id in ["Train_1", "Train_4", "Train_7"]:
                self.assertTrue(indexed.loc[image_id, "reserved_evaluation"])
                self.assertFalse(indexed.loc[image_id, "eligible_training_candidate"])
            self.assertNotEqual(indexed.loc["Train_1", "sha256"], indexed.loc["Train_4", "sha256"])
            self.assertEqual(indexed.loc["Train_1", "pixel_sha256"], indexed.loc["Train_4", "pixel_sha256"])
            self.assertFalse(indexed.loc["Train_2", "eligible_training_candidate"])
            self.assertTrue(indexed.loc["Train_2", "cedar_prior_use_unresolved"])
            self.assertEqual(indexed.loc["Train_5", "image_status"], "unreadable_image")
            self.assertEqual(indexed.loc["Train_5", "label_status"], "conflicting_or_empty_label")
            self.assertTrue(indexed.loc["Train_5", "image_error"])
            self.assertTrue(indexed.loc["Train_5", "sha256"])
            self.assertEqual(indexed.loc["Train_6", "label_status"], "exclude_multiple_diseases")
            self.assertEqual(indexed.loc["Train_8", "label_status"], "invalid_label")
            self.assertEqual(before, {str(p): audit_module.sha256(p) for p in images.iterdir()})
            with self.assertRaisesRegex(ValueError, "existing audits"):
                self.run_audit(project)

    def test_missing_manifest_and_wrong_counts_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            self.fixture(project)
            with self.assertRaisesRegex(ValueError, "Expected 40"):
                audit_module.audit(project, project / "audit")
            manifest = project / "data/healthy_structural_benchmark/manifest.csv"
            manifest.rename(manifest.with_suffix(".backup"))
            with self.assertRaisesRegex(FileNotFoundError, "Required previous-use"):
                self.run_audit(project)
            self.assertFalse((project / "audit").exists())

    def test_raw_dimensions_and_exif_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rotated.jpg"
            image = Image.new("RGB", (80, 60), "green")
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            before = path.read_bytes()
            result = audit_module.inspect_image(path)
            self.assertEqual((result["width"], result["height"]), (80, 60))
            self.assertEqual((result["oriented_width"], result["oriented_height"]), (60, 80))
            self.assertEqual(result["exif_orientation"], 6)
            self.assertTrue(result["rgb_compatible"])
            self.assertEqual(len(result["dhash256"]), 64)
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
