"""Evaluation safeguards and metrics on synthetic data; no training or benchmark inference."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from appleleaf.cli.evaluate_c1 import (
    main, paired_counts, same_inventory, score_predictions, validate_images, validate_metadata,
)
from appleleaf.config import APPLE_CLASSES


def predictions(labels, predicted):
    probabilities = np.full((len(labels), 4), .02)
    probabilities[np.arange(len(labels)), predicted] = .94
    return {"labels": np.array(labels), "predictions": np.array(predicted), "probabilities": probabilities}


class C1EvaluationTests(unittest.TestCase):
    def test_absent_black_rot_still_counts_as_error(self):
        result = score_predictions(predictions([0, 2, 3], [1, 2, 3]), [0, 2, 3])
        self.assertAlmostEqual(result["accuracy"], 2/3)
        self.assertAlmostEqual(result["macro_f1"], 2/3)
        self.assertAlmostEqual(result["macro_f1_all_four"], .5)
        self.assertEqual(result["black_rot_prediction_count"], 1)
        self.assertEqual(result["confusion_matrix"][0][1], 1)
        self.assertIsNone(result["per_class"]["Black rot"]["recall"])

    def test_confidence_bins_extremes_and_high_confidence_errors(self):
        results = predictions([0, 2, 3], [0, 1, 0])
        results["probabilities"][0] = [1, 0, 0, 0]
        metric = score_predictions(results, [0, 2, 3])
        self.assertEqual(sum(row["count"] for row in metric["confidence_bins"]), 3)
        self.assertEqual(metric["confidence"]["high_confidence_count"], 3)
        self.assertEqual(metric["confidence"]["high_confidence_errors"], 2)
        self.assertAlmostEqual(metric["confidence"]["high_confidence_error_rate"], 2/3)
        self.assertTrue(np.isfinite(metric["confidence"]["negative_log_likelihood"]))
        all_wrong = score_predictions(predictions([0, 2, 3], [1, 1, 1]), [0, 2, 3])
        self.assertIsNone(all_wrong["confidence"]["mean_correct"])
        self.assertEqual(all_wrong["confidence"]["high_confidence_error_rate"], 1)

    def test_invalid_probabilities_cannot_be_reported(self):
        result = predictions([0], [0])
        result["probabilities"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid four-way"):
            score_predictions(result, [0])

    def test_paired_improvements_and_regressions(self):
        baseline = predictions([0, 1, 2, 3], [0, 0, 2, 0])
        candidate = predictions([0, 1, 2, 3], [0, 1, 0, 0])
        self.assertEqual(paired_counts(baseline, candidate),
                         dict(both_correct=1, both_wrong=1, corrected=1, regressed=1))
        candidate["labels"][0] = 3
        with self.assertRaisesRegex(ValueError, "Paired labels"):
            paired_counts(baseline, candidate)

    def test_inventory_order_may_differ_but_images_and_labels_may_not(self):
        frame = pd.DataFrame([dict(path=f"{i}.png", label=i, class_name=APPLE_CLASSES[i], image_hash=str(i))
                              for i in [0, 1]])
        same_inventory(frame, frame.iloc[::-1])
        changed = frame.copy()
        changed.loc[0, "image_hash"] = "changed"
        with self.assertRaisesRegex(ValueError, "inventories differ"):
            same_inventory(frame, changed)

    def test_live_pixel_identity_and_overlap_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary)/"a.png", Path(temporary)/"b.png"
            image = Image.new("RGB", (8, 8), "green")
            image.save(first)
            image.save(second)
            identity = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            def frame(path, expected=identity):
                return pd.DataFrame([dict(path=str(path), label=0, class_name=APPLE_CLASSES[0], image_hash=expected)])
            with self.assertRaisesRegex(ValueError, "Pixel overlap"):
                validate_images({"pv_train": frame(first), "pv_test": frame(second)})
            with self.assertRaisesRegex(ValueError, "PV pixels changed"):
                validate_images({"pv_test": frame(first, "bad")})

    def test_existing_output_is_preserved_before_loading_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "new output directory"):
                main(["--output-dir", temporary])

    def test_wrong_model_role_or_normalization_rejected(self):
        metadata = dict(architecture="efficientnet_b0", image_size=224, preprocessing="imagenet_rgb_resize",
                        class_to_index=dict(zip(APPLE_CLASSES, range(4))), training_strategy="plantvillage_only_imagenet_transfer")
        validate_metadata(metadata, "reference")
        with self.assertRaisesRegex(ValueError, "Wrong model role"):
            validate_metadata(metadata, "robust")
        metadata["normalization"] = {"mean": [0, 0, 0], "std": [1, 1, 1]}
        with self.assertRaisesRegex(ValueError, "normalization"):
            validate_metadata(metadata, "reference")


if __name__ == "__main__":
    unittest.main()
