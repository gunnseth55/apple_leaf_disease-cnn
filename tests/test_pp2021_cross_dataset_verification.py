"""Check the independent verifier's alternate index and malformed-output refusal."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("cross_verify", Path(__file__).resolve().parents[1] / "scripts/verify_pp2021_cross_dataset.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class CrossDatasetVerificationTests(unittest.TestCase):
    def test_independent_eight_block_search_equals_brute_force(self):
        rng = np.random.default_rng(29)
        originals = [int.from_bytes(rng.bytes(32), "big") for _ in range(50)]
        four_blocks = sum(1 << shift for shift in [0, 33, 66, 99])
        values = set(originals + [value ^ four_blocks for value in originals] + [0, 15, 31])
        index = verify.EightBlockIndex(values)
        for value in originals + [0, 15, 31]:
            self.assertEqual(index.query(value), {other for other in values if (value^other).bit_count() <= 4})

    def test_incomplete_audit_and_invalid_boolean_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cross_dataset_integrity_report.json").write_text('{"complete": false}')
            with self.assertRaisesRegex(ValueError, "Incomplete audit"):
                verify.verify(root)
        with self.assertRaisesRegex(ValueError, "Invalid boolean"):
            verify.truth("maybe")


if __name__ == "__main__":
    unittest.main()
