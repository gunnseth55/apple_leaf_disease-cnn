import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from appleleaf.cli.evaluate_kashmir import FOLDER_CLASSES, build_manifest, summarize
from appleleaf.config import APPLE_CLASSES


class KashmirEvaluationTests(unittest.TestCase):
    def test_absent_class_and_four_way_predictions(self):
        results = {"labels": np.array([0, 1, 3]), "predictions": np.array([0, 2, 3]),
                   "probabilities": np.array([[.9, .03, .03, .04], [.02, .02, .94, .02], [.05, .05, .05, .85]])}
        metrics = summarize(results, list(APPLE_CLASSES))
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1_present_classes"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1_all_four_classes"], .5)
        self.assertEqual(metrics["confusion_matrix"][1][2], 1)
        self.assertEqual(metrics["per_class"][APPLE_CLASSES[2]]["support"], 0)
        self.assertEqual(metrics["confidence"]["high_confidence_error_count"], 1)
        self.assertEqual(sum(row["count"] for row in metrics["confidence"]["bins"]), 3)

    def test_conflicts_are_never_silently_scored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = {name: i for i, name in enumerate(APPLE_CLASSES)}
            for i, folder in enumerate(FOLDER_CLASSES):
                directory = root / folder
                directory.mkdir()
                Image.new("RGB", (4, 4), (i * 50, 0, 0)).save(directory / "unique.png")
            for folder in ("APPLE ROT LEAVES", "HEALTHY LEAVES"):
                Image.new("RGB", (4, 4), (200, 0, 0)).save(root / folder / "conflict.png")
            with self.assertRaisesRegex(ValueError, "conflicting class labels"):
                build_manifest(root, mapping)
            frame, rejected, excluded = build_manifest(root, mapping, True)
            self.assertEqual(len(frame), 3)
            self.assertEqual(len(excluded), 2)
            self.assertEqual(rejected, [])

    def test_missing_lfs_contents_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i, folder in enumerate(FOLDER_CLASSES):
                directory = root / folder
                directory.mkdir()
                Image.new("RGB", (4, 4), (i * 50, 0, 0)).save(directory / "real.png")
            (root / "SCAB LEAVES" / "missing.jpeg").write_text(
                "version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 999\n")
            frame, rejected, excluded = build_manifest(root, {n: i for i, n in enumerate(APPLE_CLASSES)})
            self.assertEqual(len(frame), 3)
            self.assertEqual(rejected[0]["reason"], "missing_git_lfs_image")
            self.assertTrue(excluded.empty)


if __name__ == "__main__":
    unittest.main()
