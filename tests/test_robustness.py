"""Data and configuration tests; no training epochs or model inference."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from appleleaf.cli.evaluate_kashmir import select_subset, summarize
from appleleaf.cli.train_efficientnet import parse_args
from appleleaf.config import APPLE_CLASSES
from appleleaf.datasets import BackgroundTestDataset
from appleleaf.robustness import RobustBackgroundDataset, RobustImageTransform, deterministic_texture


class RobustnessTests(unittest.TestCase):
    def test_training_replacement_preserves_all_mask_foreground_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = np.full((20, 20, 3), 80, dtype=np.uint8)
            pixels[5:15, 5:15] = (120, 180, 25)
            mask = np.zeros((20, 20, 3), dtype=np.uint8)
            mask[5:15, 5:15] = 255
            Image.fromarray(pixels).save(root / "image.png")
            Image.fromarray(mask).save(root / "mask.png")
            frame = pd.DataFrame([{"path": str(root / "image.png"),
                                   "segmented_path": str(root / "mask.png"), "label": 0}])
            dataset = RobustBackgroundDataset(frame)
            with patch("appleleaf.robustness.torch.rand", return_value=torch.tensor(.9)):
                augmented = np.asarray(dataset.make_image(0))
            np.testing.assert_array_equal(augmented[5:15, 5:15], pixels[5:15, 5:15])
            self.assertFalse(np.array_equal(augmented[:5], pixels[:5]))
            # Evaluation texture is deterministic and also preserves the leaf.
            evaluation = BackgroundTestDataset(frame, "smooth_texture", np.asarray)
            first, _ = evaluation[0]
            second, _ = evaluation[0]
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first[5:15, 5:15], pixels[5:15, 5:15])

    def test_transform_reproducibility_shape_and_finiteness(self):
        image = Image.new("RGB", (100, 170), (80, 160, 40))
        transform = RobustImageTransform()
        torch.manual_seed(42)
        first = transform(image)
        torch.manual_seed(42)
        second = transform(image)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(tuple(first.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(first).all())
        # The test pattern does not consume stochastic training generator state.
        state = torch.get_rng_state().clone()
        deterministic_texture((20, 20), "example")
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_scab_healthy_subset_keeps_out_of_subset_predictions_as_errors(self):
        frame = pd.DataFrame({"class_name": [APPLE_CLASSES[i] for i in (0, 1, 3)],
                              "label": [0, 1, 3]})
        selected, excluded = select_subset(frame, "scab-healthy")
        self.assertEqual(selected.label.tolist(), [0, 3])
        self.assertEqual(excluded.label.tolist(), [1])
        result = {"labels": np.array([0, 3]), "predictions": np.array([2, 1]),
                  "probabilities": np.array([[.1, .1, .7, .1], [.1, .7, .1, .1]])}
        report = summarize(result, list(APPLE_CLASSES))
        self.assertEqual(report["accuracy"], 0)
        self.assertEqual(report["macro_f1_present_classes"], 0)
        self.assertEqual(report["confusion_matrix"][0][2], 1)
        self.assertEqual(report["confusion_matrix"][3][1], 1)

    def test_robust_profile_protects_reference_folder(self):
        with self.assertRaises(SystemExit):
            parse_args(["--dataset-path", "unused", "--augmentation-profile", "robust",
                        "--artifacts-dir", "artifacts/efficientnet_b0"])
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(["--dataset-path", "unused", "--augmentation-profile", "robust",
                               "--artifacts-dir", str(Path(temporary) / "new_run"), "--dry-run"])
            self.assertEqual((args.epochs, args.warmup_epochs, args.seed), (15, 3, 42))


if __name__ == "__main__":
    unittest.main()
