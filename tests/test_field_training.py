"""No optimization: check mixed-source sampling, metric scope and run safeguards."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from appleleaf.cli.train_field import parse_args, sampling_weights, source_metrics


class FieldTrainingTests(unittest.TestCase):
    def test_source_and_class_probability_mass(self):
        frame = pd.DataFrame([{"source":"plantvillage", "label":label} for label in [0,0,0,1,2,3]]
                             + [{"source":"pp2020", "label":label} for label in [0,2,2,3,3,3]])
        weights = sampling_weights(frame).numpy()
        self.assertAlmostEqual(weights.sum(), 1.)
        for source in ["plantvillage", "pp2020"]:
            selected = frame.source == source
            self.assertAlmostEqual(weights[selected].sum(), .5)
            for label in frame[selected].label.unique():
                self.assertAlmostEqual(weights[selected & frame.label.eq(label)].sum(), .5/frame[selected].label.nunique())
        custom = sampling_weights(frame, .25).numpy()
        self.assertAlmostEqual(custom[frame.source.eq("pp2020")].sum(), .25)

    def test_missing_source_cannot_silently_train_single_source(self):
        with self.assertRaisesRegex(ValueError, "Missing training source"):
            sampling_weights(pd.DataFrame([{"source":"plantvillage", "label":0}]))

    def test_field_metric_does_not_penalize_absent_class_as_support(self):
        results = dict(labels=np.array([0,2,3]), predictions=np.array([0,2,3]))
        field = source_metrics(results, [0,2,3])
        self.assertEqual(field["macro_f1"], 1.)
        self.assertEqual(field["macro_f1_all_four"], .75)
        results["predictions"][0] = 1
        self.assertLess(source_metrics(results, [0,2,3])["macro_f1"], 1.)

    def test_existing_run_and_invalid_rates_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "run"
            existing.mkdir()
            with self.assertRaises(SystemExit):
                parse_args(["--artifacts-dir", str(existing)])
            with self.assertRaises(SystemExit):
                parse_args(["--backbone-lr", "nan"])
            args = parse_args(["--dry-run", "--artifacts-dir", str(Path(temporary)/"new")])
            self.assertTrue(args.dry_run)
            self.assertFalse(args.artifacts_dir.exists())


if __name__ == "__main__":
    unittest.main()
