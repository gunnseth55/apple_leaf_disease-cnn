"""Preserve held-out aliases and distinguish classifier training from pipeline novelty."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("pp2020_role_preservation", SCRIPTS / "preserve_pp2020_heldout.py")
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
try:
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


class RolePreservationTests(unittest.TestCase):
    def group(self, status, roles):
        return pd.DataFrame([dict(status=status, member_role=role) for role in roles])

    def test_heldout_aliases_never_enter_training_or_new_validation(self):
        policy = module.group_permissions(self.group("reserved_evaluation", ["labelled_source", "evaluation_only"]))
        self.assertTrue(policy["reserved_existing_evaluation"])
        self.assertFalse(policy["eligible_classification_training"])
        self.assertFalse(policy["eligible_new_pipeline_validation"])
        self.assertFalse(policy["eligible_new_unseen_pipeline_test"])
        self.assertTrue(policy["eligible_existing_evaluation"])

    def test_segmentation_training_can_train_classifier_but_is_not_unseen(self):
        policy = module.group_permissions(self.group("accepted_previously_trained", ["labelled_source", "segmentation_training"]))
        self.assertTrue(policy["eligible_classification_training"])
        self.assertFalse(policy["eligible_new_pipeline_validation"])
        self.assertFalse(policy["eligible_new_unseen_pipeline_test"])
        self.assertTrue(policy["previously_seen_by_segmenter"])

    def test_quarantine_and_conflicting_history_override_permissions(self):
        for status, roles in [("quarantine_label_conflict", ["labelled_source"]),
                              ("excluded_from_c1", ["labelled_source"]),
                              ("accepted", ["segmentation_training", "evaluation_only"])]:
            policy = module.group_permissions(self.group(status, roles))
            self.assertFalse(policy["eligible_classification_training"])
            self.assertFalse(policy["eligible_existing_evaluation"])

    def test_validator_catches_file_alias_and_cross_group_bypasses(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = []
            for group_id, status, roles in [("held", "reserved_evaluation", ["labelled_source", "evaluation_only"]),
                                             ("seen", "accepted_previously_trained", ["labelled_source", "segmentation_training"]),
                                             ("new", "accepted", ["labelled_source", "copy"])]:
                policy = module.group_permissions(self.group(status, roles))
                for index, role in enumerate(roles):
                    rows.append(dict(group_id=group_id, image_path=str(Path(temporary) / f"{group_id}_{index}.jpg"), **policy))
            registry = pd.DataFrame(rows)
            for path in registry[registry.group_id == "held"].image_path:
                with self.assertRaises(ValueError):
                    module.validate_proposed_roles(pd.DataFrame([dict(image_path=path, role="train")]), registry)
            with self.assertRaises(ValueError):
                module.validate_proposed_roles(pd.DataFrame([dict(group_id="seen", role="test")]), registry)
            with self.assertRaises(ValueError):
                module.validate_proposed_roles(pd.DataFrame([dict(group_id="new", role="train"), dict(group_id="new", role="test")]), registry)
            with self.assertRaises(ValueError):
                module.validate_proposed_roles(pd.DataFrame([dict(image_path=registry.image_path.iloc[0], group_id="new", role="train")]), registry)
            module.validate_proposed_roles(pd.DataFrame([dict(group_id="seen", role="train"), dict(group_id="held", role="existing_evaluation"), dict(group_id="new", role="test")]), registry)


if __name__ == "__main__":
    unittest.main()
