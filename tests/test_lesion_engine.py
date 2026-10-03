import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from appleleaf_lesion.engine import calibrate_threshold, metrics_from_totals


class _CalibrationDataset(Dataset):
    def __init__(self):
        self.images = torch.tensor([0.6, 0.4, 0.3, 0.1]).view(-1, 1, 1, 1)
        self.masks = torch.tensor([1.0, 1.0, 0.0, 0.0]).view(-1, 1, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.masks[index], str(index)


class _ProbabilityModel(torch.nn.Module):
    def forward(self, probabilities):
        return torch.logit(probabilities.clamp(1e-6, 1 - 1e-6))


class LesionEngineTests(unittest.TestCase):
    def test_metrics_from_totals(self):
        metrics = metrics_from_totals(3, 1, 2)
        self.assertAlmostEqual(metrics["dice"], 6 / 9)
        self.assertAlmostEqual(metrics["iou"], 3 / 6)
        self.assertAlmostEqual(metrics["precision"], 3 / 4)
        self.assertAlmostEqual(metrics["recall"], 3 / 5)

    def test_calibrate_threshold_uses_best_validation_dice(self):
        loader = DataLoader(_CalibrationDataset(), batch_size=2, shuffle=False)
        threshold, metrics = calibrate_threshold(
            _ProbabilityModel(), loader, torch.device("cpu"), thresholds=[0.3, 0.4, 0.5]
        )
        self.assertEqual(threshold, 0.4)
        self.assertAlmostEqual(metrics["dice"], 1.0)


if __name__ == "__main__":
    unittest.main()
