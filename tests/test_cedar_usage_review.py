"""Usage-evidence rules must not confuse copying, manifests and executed inference."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("cedar_usage_review", SCRIPTS / "review_pp2020_cedar_usage.py")
review = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
try:
    spec.loader.exec_module(review)
finally:
    sys.path.pop(0)


class CedarUsageReviewTests(unittest.TestCase):
    def test_recovered_image_resolves_lfs_placeholder(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pointer = directory / "pointer.jpg"
            recovered = directory / "recovered.jpg"
            pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
            recovered.write_bytes(b"image content")
            self.assertEqual(review.resolve_record_image([pointer, recovered]), recovered)
            self.assertIsNone(review.resolve_record_image([pointer]))

    def test_copying_alone_does_not_contaminate(self):
        status, allowed = review.summarize_roles(set())
        self.assertEqual(status, "copy_verified_no_other_use_found_in_reviewed_records")
        self.assertIn("future_train_or_validation_or_test", allowed)
        self.assertIn("assign_one_role_only", allowed)

    def test_development_or_evaluation_prevents_fresh_test_claims(self):
        for roles in [{"trained"}, {"model_selection_membership"}, {"evaluated"}]:
            _, allowed = review.summarize_roles(roles)
            self.assertIn("not_fresh_independent_test", allowed)
        status, allowed = review.summarize_roles({"validation_manifest_membership_unconfirmed"})
        self.assertEqual(status, "prior_use_requires_review")
        self.assertIn("hold", allowed)

    def test_manifest_alone_does_not_prove_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.assertEqual(review.csv_role(folder / "manifest.csv", ["image_id"]), "manifest_membership_unconfirmed")
            self.assertEqual(review.csv_role(folder / "train_manifest.csv", ["image_id"]), "training_manifest_membership_unconfirmed")
            (folder / "training_history.csv").write_text("epoch\n1\n")
            (folder / "checkpoint.pt").touch()
            self.assertEqual(review.csv_role(folder / "train_manifest.csv", ["image_id"]), "trained")
            self.assertEqual(review.csv_role(folder / "validation_manifest.csv", ["image_id"]), "model_selection_membership")
            self.assertEqual(review.csv_role(folder / "predictions.csv", ["image_id", "predicted_class"]), "evaluated")
            self.assertEqual(review.csv_role(folder / "excluded_images.csv", ["image_id"]), "reference_only")
        self.assertEqual(set(review.ID_RE.findall("Train_1.jpg Train_12.jpg")), {"Train_1", "Train_12"})


if __name__ == "__main__":
    unittest.main()
