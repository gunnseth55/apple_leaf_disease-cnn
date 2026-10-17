"""Frozen-split reproducibility, quotas, duplicate canonicalization and refusal rules."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("pp2020_field_splits", SCRIPTS / "build_pp2020_field_splits.py")
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
try:
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


class FieldSplitTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for label, count in [(0, 591), (2, 622), (3, 466)]:
            for index in range(count):
                rows.append(dict(group_id=f"g{label}_{index:04d}", label=label,
                                 previously_seen_by_segmenter=label == 3 and index < 40))
        return pd.DataFrame(rows)

    def test_integer_targets_conserve_all_images(self):
        self.assertEqual(module.targets(591), dict(train=414, val=88, test=89))
        self.assertEqual(module.targets(622), dict(train=436, val=93, test=93))
        self.assertEqual(module.targets(466), dict(train=326, val=70, test=70))
        for count in range(1, 100):
            self.assertEqual(sum(module.targets(count).values()), count)

    def test_seed_row_order_and_mandatory_training(self):
        frame = self.fixture()
        assignment, quotas = module.allocate_groups(frame, 42)
        replay, _ = module.allocate_groups(frame.sample(frac=1, random_state=7), 42)
        self.assertEqual(assignment, replay)
        changed, _ = module.allocate_groups(frame, 43)
        self.assertNotEqual(assignment, changed)
        frame["split"] = frame.group_id.map(assignment)
        self.assertTrue(frame.loc[frame.previously_seen_by_segmenter, "split"].eq("train").all())
        for label in [0, 2, 3]:
            counts = frame[frame.label == label].split.value_counts().to_dict()
            self.assertEqual(counts, quotas[str(label)])

    def test_group_is_not_split_even_with_multiple_rows(self):
        frame = pd.DataFrame([dict(group_id=group, label=0, previously_seen_by_segmenter=False)
                              for group in ["A", "A", "A", "B", "C", "D", "E"]])
        assignment, _ = module.allocate_groups(frame, 42)
        self.assertEqual(set(assignment), set(frame.group_id))
        self.assertEqual(frame.assign(split=frame.group_id.map(assignment)).groupby("group_id").split.nunique().max(), 1)

    def test_canonicalization_excludes_copy_and_redundant_original(self):
        rows = []
        for image_id in ["Train_1", "Train_2"]:
            rows.append(dict(image_id=image_id, group_id="same_photo", image_path=f"/{image_id}.jpg",
                             eligible_classification_training=True, label_name="Apple___Apple_scab", label=0,
                             sha256="same", previously_seen_by_segmenter=False,
                             reserved_existing_evaluation=False, blocked_by_adjudication=False, status="accepted"))
        joined = pd.DataFrame(rows)
        members = pd.DataFrame([dict(image_id=row["image_id"], group_id="same_photo", image_path=row["image_path"], source_record=True, member_role="labelled_source") for row in rows]
                               + [dict(image_id="Train_1", group_id="same_photo", image_path="/copy.jpg", source_record=False, member_role="cedar_copy")])
        inventory = pd.DataFrame([dict(image_id=row["image_id"], pixel_sha256="pixels") for row in rows])
        frames, _ = module.build_frames(joined, members, inventory, 42)
        accepted = pd.concat([frames[name+".csv"] for name in module.SPLITS])
        self.assertEqual(accepted.image_id.tolist(), ["Train_1"])
        self.assertEqual(len(frames["excluded.csv"]), 2)
        self.assertTrue(frames["excluded.csv"].inherited_group_split.eq(accepted.split.iloc[0]).all())
        joined.loc[1, "label_name"] = "Apple___healthy"
        with self.assertRaisesRegex(ValueError, "Mixed-label"):
            module.build_frames(joined, members, inventory, 42)

    def test_no_overwrite_or_unversioned_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            frozen = project / "data/field_manifests/pp2020_split_v1"
            frozen.mkdir(parents=True)
            (frozen / "sentinel").write_text("preserve")
            with self.assertRaisesRegex(ValueError, "already exists"):
                module.run(project, frozen, seed=43)
            self.assertEqual((frozen / "sentinel").read_text(), "preserve")
            with self.assertRaisesRegex(ValueError, "explicit version"):
                module.run(project, frozen.parent / "arbitrary_name")
            with self.assertRaisesRegex(ValueError, "Required input missing"):
                module.run(project, frozen.parent / "pp2020_split_v2")
            self.assertFalse((frozen.parent / "pp2020_split_v2").exists())

    def test_missing_or_invalid_permissions_are_not_truthy_strings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "roles.csv"
            for value in ["", "unknown", "1"]:
                pd.DataFrame([dict(eligible_classification_training=value)]).to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, "boolean"):
                    module.read_csv(path)


if __name__ == "__main__":
    unittest.main()
