"""Independently check saved C1 comparison outputs; no model loading or inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify(root):
    root = Path(root)
    completion = json.loads((root / "completion.json").read_text())
    assert completion["complete"] and completion["all_frozen_inputs_unchanged"]
    validation = json.loads((root / "input_validation.json").read_text())
    for path, expected in validation["input_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    comparison = pd.read_csv(root / "comparison.csv")
    checks = []
    for source, present, expected in [("plantvillage", [0, 1, 2, 3], 475), ("pp2020", [0, 2, 3], 252)]:
        manifest_path = root / f"{source}_test_manifest.csv"
        manifest = pd.read_csv(manifest_path)
        paired = pd.read_csv(root / f"{source}_paired_predictions.csv")
        assert len(manifest) == expected and paired.path.equals(manifest.path)
        for model in ["reference", "robust", "c1"]:
            folder = root / model / source
            frame = pd.read_csv(folder / "predictions.csv")
            metrics = json.loads((folder / "metrics.json").read_text())
            assert metrics["metadata"]["test_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            with np.load(folder / "predictions.npz") as saved:
                labels, predicted, probabilities = [saved[key] for key in ["labels", "predictions", "probabilities"]]
            assert np.array_equal(labels, manifest.label) and frame.path.equals(manifest.path)
            assert np.array_equal(frame.prediction, predicted) and np.array_equal(paired[f"{model}_prediction"], predicted)
            assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
            assert np.array_equal(probabilities.argmax(axis=1), predicted)
            probability_columns = [column for column in frame if column.startswith("probability_")]
            assert np.allclose(frame[probability_columns], probabilities, atol=1e-12, rtol=1e-12)
            correct = labels == predicted
            assert np.array_equal(frame.correct, correct)
            matrix = np.zeros((4, 4), dtype=int)
            np.add.at(matrix, (labels, predicted), 1)
            assert np.array_equal(matrix, metrics["confusion_matrix"])
            assert np.array_equal(matrix, pd.read_csv(folder / "confusion_matrix.csv", index_col=0).to_numpy())
            per_class = pd.read_csv(folder / "per_class_metrics.csv", index_col=0)
            f1 = np.divide(2*matrix.diagonal(), matrix.sum(axis=0)+matrix.sum(axis=1),
                           out=np.zeros(4), where=matrix.sum(axis=0)+matrix.sum(axis=1) != 0)
            accuracy, macro_f1 = float(correct.mean()), float(f1[present].mean())
            assert np.isclose(accuracy, metrics["accuracy"]) and np.isclose(macro_f1, metrics["macro_f1"])
            assert np.isclose(f1.mean(), metrics["macro_f1_all_four"])
            assert np.allclose(f1, per_class.f1_score if "f1_score" in per_class else per_class["f1-score"])
            for index in range(4):
                if matrix[index].sum():
                    assert np.isclose(per_class.iloc[index].recall, matrix[index, index] / matrix[index].sum())
                else:
                    assert pd.isna(per_class.iloc[index].recall)
            row = comparison[comparison.model.eq(model) & comparison.source.eq(source)].iloc[0]
            assert np.isclose(row.accuracy, accuracy) and np.isclose(row.macro_f1, macro_f1)
            confidence = probabilities.max(axis=1)
            high = confidence >= .9
            assert metrics["confidence"]["high_confidence_errors"] == int((high & ~correct).sum())
            assert metrics["confidence"]["high_confidence_count"] == int(high.sum())
            bins = pd.read_csv(folder / "confidence_bins.csv")
            assert bins["count"].sum() == expected and bins.correct.sum() == int(correct.sum())
            assert len(pd.read_csv(folder / "errors.csv")) == int((~correct).sum())
            reproduction = None
            if source == "plantvillage" and model != "c1":
                training_dir = Path("artifacts") / ("efficientnet_b0" if model == "reference" else "efficientnet_b0_robust")
                previous = pd.read_csv(training_dir / "test_predictions.csv").set_index("path").loc[frame.path]
                assert np.array_equal(previous.prediction, predicted)
                assert np.array_equal(previous.label, labels)
                reproduction = {"all_475_original_predictions_match": True,
                                "max_probability_difference": float(np.abs(previous[probability_columns].to_numpy()-probabilities).max())}
            checks.append({"model": model, "source": source, "sample_count": expected,
                           "accuracy": accuracy, "macro_f1": macro_f1, "passed": True,
                           "prior_pv_reproduction": reproduction})
    for row in pd.read_csv(root / "paired_comparison.csv").itertuples():
        frame = pd.read_csv(root / f"{row.source}_paired_predictions.csv")
        left = frame[f"{row.baseline}_prediction"] == frame.label
        right = frame[f"{row.candidate}_prediction"] == frame.label
        assert row.corrected == int((~left & right).sum()) and row.regressed == int((left & ~right).sum())
        assert row.both_correct == int((left & right).sum()) and row.both_wrong == int((~left & ~right).sum())
        assert np.isclose(row.delta_accuracy, right.mean()-left.mean())
    return {"passed": True, "method": "Saved CSV/NPZ alignment; manual confusion-matrix accuracy/F1/recall; confidence counts; paired changes; input SHA-256; prior PV prediction parity",
            "model_inference_performed": False, "checks": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/c1_heldout_comparison_v1"))
    args = parser.parse_args()
    result = verify(args.root)
    print(json.dumps(result, indent=2))
