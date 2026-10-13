"""Check demo preprocessing, mask mapping, and independent model outputs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from appleleaf.datasets import checkpoint_transform
from appleleaf.demo import DemoPipeline, PRIMARY, lesion_overlay, segmentation_input, prepare_crop, seed_crop
from appleleaf_lesion.dataset import LesionDataset


class FixedModel(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.input_shape = None
        self.inputs = None

    def forward(self, inputs):
        self.input_shape = tuple(inputs.shape)
        self.inputs = inputs.detach().clone()
        return self.output


class DemoTests(unittest.TestCase):
    def test_crop_uses_applied_composite_and_rejects_missing_crop(self):
        original = Image.fromarray(np.random.default_rng(8).integers(0, 256, (90, 120, 3), dtype=np.uint8))
        crop = original.crop((10, 15, 80, 75))
        selected = prepare_crop({"background": original, "layers": [], "composite": crop})
        np.testing.assert_array_equal(np.asarray(selected), np.asarray(crop))
        self.assertEqual(selected.size, (70, 60))
        np.testing.assert_array_equal(np.asarray(seed_crop(original)["composite"]), np.asarray(original))
        for invalid in [None, {}, {"background": original, "composite": None}]:
            with self.assertRaises(ValueError):
                prepare_crop(invalid)

    def test_segmentation_matches_existing_evaluation_preprocessing(self):
        image = Image.fromarray(np.random.default_rng(5).integers(0, 256, (73, 111, 3), dtype=np.uint8))
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            image.save(folder / "image.png")
            Image.new("L", image.size).save(folder / "mask.png")
            pd.DataFrame([{"image_path": str(folder / "image.png"),
                           "mask_path": str(folder / "mask.png")}]).to_csv(folder / "manifest.csv", index=False)
            expected, _, _ = LesionDataset(folder / "manifest.csv")[0]
            torch.testing.assert_close(segmentation_input(image), expected, rtol=0, atol=0)

    def test_overlay_preserves_original_and_binary_mask_alignment(self):
        image = Image.new("RGB", (8, 4), (20, 40, 60))
        probability = np.array([[0.69, 0.70], [0.1, 0.9]])
        overlay, mask = lesion_overlay(image, probability, 0.70)
        self.assertEqual(overlay.size, image.size)
        self.assertEqual(set(np.unique(np.asarray(mask))), {0, 255})
        np.testing.assert_array_equal(np.array(overlay)[:, :4], np.array(image)[:, :4])
        self.assertTrue(np.all(np.array(overlay)[:, 4:, 0] > 20))
        self.assertEqual(image.getpixel((7, 3)), (20, 40, 60))

    def test_healthy_prediction_does_not_hide_segmentation(self):
        pipeline = DemoPipeline.__new__(DemoPipeline)
        pipeline.device = torch.device("cpu")
        pipeline.classifier = FixedModel(torch.tensor([[0., 0., 0., 5.]]))
        pipeline.labels = ["Apple scab", "Black rot", "Cedar apple rust", "healthy"]
        pipeline.classifier_transform = checkpoint_transform({"image_size": 224, "preprocessing": "imagenet_rgb_resize"})
        segmenter = FixedModel(torch.full((1, 1, 256, 256), 2.))
        pipeline.segmenters = {PRIMARY: (segmenter, 0.70)}
        scores, original, overlay, mask, details = pipeline.predict(Image.new("RGB", (101, 67)), PRIMARY)
        self.assertEqual(max(scores, key=scores.get), "healthy")
        self.assertAlmostEqual(sum(scores.values()), 1., places=6)
        self.assertEqual(pipeline.classifier.input_shape, (1, 3, 224, 224))
        self.assertEqual(segmenter.input_shape, (1, 3, 256, 256))
        self.assertEqual(original.size, overlay.size)
        self.assertTrue(np.all(np.array(mask) == 255))
        self.assertIn("0.70", details)
        self.assertIn("Warning: model disagreement", details)
        self.assertIn("not a disease diagnosis", details)
        source = Image.fromarray(np.random.default_rng(9).integers(0, 256, (100, 150, 3), dtype=np.uint8))
        crop = source.crop((25, 20, 110, 80))
        editor = {"background": source, "layers": [], "composite": crop}
        paired = pipeline.compare_crop(source, editor)
        np.testing.assert_array_equal(np.asarray(paired[1]), np.asarray(crop))
        torch.testing.assert_close(pipeline.classifier.inputs[0], pipeline.classifier_transform(crop), rtol=0, atol=0)
        torch.testing.assert_close(segmenter.inputs[0], segmentation_input(crop), rtol=0, atol=0)
        self.assertEqual(paired[2].size, crop.size)
        self.assertEqual(paired[3].size, crop.size)
        self.assertIn("Full-image softmax score", paired[5])
        self.assertIn("not diagnostic certainty", paired[4])
        segmenter.output.fill_(-2.)
        _, _, _, clean_mask, clean_details = pipeline.predict(original)
        self.assertTrue(np.all(np.array(clean_mask) == 0))
        self.assertNotIn("Warning: model disagreement", clean_details)
        self.assertIn("does not establish", clean_details)
        with self.assertRaisesRegex(ValueError, "Upload"):
            pipeline.predict(None)
        with self.assertRaisesRegex(ValueError, "available"):
            pipeline.predict(original, "missing")


if __name__ == "__main__":
    unittest.main()
