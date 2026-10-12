"""Compare paired saved reports, including manifest identity checks; no inference."""

import argparse
import json
from pathlib import Path

import pandas as pd


def assert_same_manifest(left, right, columns):
    a, b = pd.read_csv(left), pd.read_csv(right)
    if not a[columns].equals(b[columns]):
        raise ValueError(f"Comparison inventories differ: {left} vs {right}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    assert_same_manifest(root / "reference/plantvillage/predictions.csv",
                         root / "robust/plantvillage/predictions.csv", ["path", "label", "image_hash"])
    assert_same_manifest(root / "reference/background_audit.manifest.csv",
                         root / "robust/background_audit.manifest.csv", ["path", "label", "image_hash"])
    assert_same_manifest(root / "reference/kashmir_scab_healthy/manifest.csv",
                         root / "robust/kashmir_scab_healthy/manifest.csv", ["relative_path", "label", "sha256"])
    rows = []
    for model in ("reference", "robust"):
        directory = root / model
        assert_same_manifest(directory / "plantvillage/predictions.csv",
                             directory / "background_audit.manifest.csv", ["path", "label", "image_hash"])
        pv = json.loads((directory / "plantvillage/metrics.json").read_text())
        kashmir = json.loads((directory / "kashmir_scab_healthy/metrics.json").read_text())
        bg = pd.read_csv(directory / "background_audit.csv")
        details = json.loads((directory / "background_audit.details.json").read_text())
        if details["metadata"]["split"] != "test":
            raise ValueError("Comparison requires test-split background audits")
        hashes = (pv["metadata"]["checkpoint_sha256"], kashmir["metadata"]["checkpoint_sha256"],
                  details["metadata"]["checkpoint_sha256"])
        if len(set(hashes)) != 1:
            raise ValueError("Evaluations for one model used different checkpoint weights")
        if kashmir["metadata"]["subset"] != "scab-healthy":
            raise ValueError("Expected scab/healthy-only external support")
        for benchmark, report, class_keys in (
            ("PlantVillage test", pv, ("Apple scab", "healthy")),
            ("Kashmir scab/healthy (preliminary)", kashmir, ("Apple___Apple_scab", "Apple___healthy")),
        ):
            rows.append({"model": model, "benchmark": benchmark, "accuracy": report["accuracy"],
                         "macro_f1": report.get("macro_f1", report.get("macro_f1_present_classes")),
                         "scab_recall": report["per_class"][class_keys[0]]["recall"],
                         "healthy_recall": report["per_class"][class_keys[1]]["recall"]})
        for _, row in bg.iterrows():
            classes = details[row.background]["per_class"]
            rows.append({"model": model, "benchmark": f"Background test: {row.background}",
                         "accuracy": row.accuracy, "macro_f1": row.macro_f1,
                         "scab_recall": classes["Apple scab"]["recall"],
                         "healthy_recall": classes["healthy"]["recall"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "comparison.csv", index=False)
    reference = frame[frame.model == "reference"].set_index("benchmark")
    robust = frame[frame.model == "robust"].set_index("benchmark")
    delta = robust.select_dtypes("number") - reference.select_dtypes("number")
    delta.to_csv(root / "robust_minus_reference.csv")
    print(frame.to_string(index=False))
    print("\nRobust minus reference (proportions, not percentage points):")
    print(delta.to_string())
    print("Kashmir remains evaluation-only and preliminary. No automatic deployment promotion.")


if __name__ == "__main__":
    main()
