"""Exact identity, dHash review, group conflicts and split-lock constraints."""

import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("duplicate_adjudication", SCRIPTS / "adjudicate_pp2020_duplicates.py")
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
try:
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


class DuplicateAdjudicationTests(unittest.TestCase):
    def test_exact_pixels_vs_perceptual_candidate(self):
        left = dict(image_status="ok", pixel_sha256="a"*64, dhash256="0"*64)
        exact = {**left}
        self.assertEqual(module.judge_pair(left, exact)[0], "confirmed_duplicate")
        similar = dict(image_status="ok", pixel_sha256="b"*64, dhash256="0"*63+"1")
        self.assertEqual(module.judge_pair(left, similar)[0], "needs_visual_review")
        decision = dict(decision="false_positive", reason="Different leaf veins and lesion positions", reviewer="visual_review")
        self.assertEqual(module.judge_pair(left, similar, decision)[0], "false_positive")
        with self.assertRaises(ValueError):
            module.judge_pair(left, similar, dict(decision="confirmed_duplicate"))

    def test_only_confirmed_links_form_transitive_groups(self):
        graph = module.Components(["a", "b", "c", "d"])
        graph.join("a", "b")
        graph.join("b", "c")
        self.assertEqual(graph.find("a"), graph.find("c"))
        self.assertNotEqual(graph.find("a"), graph.find("d"))

    def test_multiple_disease_label_is_not_ignored_in_conflict(self):
        scab, multiple, healthy = (0, 0, 0, 1), (0, 1, 0, 0), (1, 0, 0, 0)
        self.assertEqual(module.restriction({scab, multiple}, {"labelled_source"})[0], "quarantine_label_conflict")
        self.assertEqual(module.restriction({multiple}, {"labelled_source"})[0], "excluded_from_c1")
        self.assertEqual(module.restriction({healthy}, {"evaluation_only"})[0], "reserved_evaluation")
        self.assertEqual(module.restriction({healthy}, {"segmentation_training"})[0], "accepted_previously_trained")

    def test_split_assignments_cannot_bypass_group_restrictions(self):
        members = pd.DataFrame([
            dict(group_id="A", image_id="Train_1", status="accepted"),
            dict(group_id="A", image_id="Train_2", status="accepted"),
            dict(group_id="B", image_id="Train_3", status="reserved_evaluation"),
            dict(group_id="C", image_id="Train_4", status="quarantine_label_conflict"),
            dict(group_id="D", image_id="Train_5", status="accepted_previously_trained"),
        ])
        for rows in [[("A", "train"), ("A", "test")], [("B", "train")], [("C", "test")], [("D", "test")], [("missing", "train")]]:
            with self.assertRaises(ValueError):
                module.enforce_split_constraints(pd.DataFrame(rows, columns=["group_id", "split"]), members)
        module.enforce_split_constraints(pd.DataFrame([("A", "train"), ("B", "test"), ("D", "train")], columns=["group_id", "split"]), members)


if __name__ == "__main__":
    unittest.main()
