"""Inference and configuration checks only: no training epochs or optimizer steps."""

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from appleleaf.cli.train_efficientnet import parse_args, save_best, set_training_phase
from appleleaf.config import APPLE_CLASSES
from appleleaf.datasets import checkpoint_transform
from appleleaf.model import build_model
from appleleaf.workflow import load_model


class TransferLearningTests(unittest.TestCase):
    def test_checkpoint_preprocessing_and_legacy_default(self):
        image = Image.new("RGB", (80, 100), "white")
        legacy = checkpoint_transform({"image_size": 128})(image)
        self.assertEqual(tuple(legacy.shape), (3, 128, 128))
        self.assertTrue(torch.equal(legacy, torch.ones_like(legacy)))
        transfer = checkpoint_transform({"image_size": 224,
                                         "preprocessing": "imagenet_rgb_resize"})(image)
        expected = (torch.ones(3) - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
        self.assertEqual(tuple(transfer.shape), (3, 224, 224))
        torch.testing.assert_close(transfer[:, 0, 0], expected)

    def test_efficientnet_checkpoint_roundtrip_without_weight_download(self):
        model = build_model("efficientnet_b0", pretrained=False).eval()
        set_training_phase(model, warmup=True)
        self.assertTrue(all(not p.requires_grad for p in model.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        set_training_phase(model, warmup=False)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        inputs = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            expected = model(inputs)
        self.assertEqual(tuple(expected.shape), (1, 4))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            save_best(path, model, {"architecture": "efficientnet_b0", "image_size": 224,
                                   "preprocessing": "imagenet_rgb_resize",
                                   "class_to_index": dict(zip(APPLE_CLASSES, range(4)))})
            restored, _ = load_model(path, torch.device("cpu"))
            with torch.no_grad():
                torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)

    def test_legacy_checkpoint_still_loads(self):
        model = build_model().eval()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.pt"
            save_best(path, model, {"class_to_index": dict(zip(APPLE_CLASSES, range(4)))})
            restored, _ = load_model(path, torch.device("cpu"))
            inputs = torch.zeros(1, 3, 128, 128)
            with torch.no_grad():
                torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)

    def test_training_arguments_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            argv = ["--dataset-path", temporary, "--artifacts-dir", str(output)]
            args = parse_args(argv)
            self.assertEqual((args.epochs, args.warmup_epochs), (15, 3))
            with self.assertRaises(SystemExit):
                parse_args(argv + ["--epochs", "3"])
            output.mkdir()
            (output / "keep.txt").write_text("existing result")
            with self.assertRaises(SystemExit):
                parse_args(argv)


if __name__ == "__main__":
    unittest.main()
