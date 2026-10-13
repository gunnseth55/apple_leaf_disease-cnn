"""Five fixed cases through a real local Gradio upload server; no model tuning."""

import argparse
import hashlib
import json
from pathlib import Path

from gradio_client import Client, handle_file
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import functional, InterpolationMode

from appleleaf.demo import (
    DEFAULT_CLASSIFIER, DEFAULT_SEGMENTER, DEFAULT_COMPARISON, PRIMARY,
    DemoPipeline, build_demo, prepare_image,
)


def expected_tensor(image, size, interpolation):
    return functional.normalize(
        functional.to_tensor(functional.resize(image, [size, size], interpolation=interpolation)),
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    ).unsqueeze(0)


def select_cases():
    pv = pd.read_csv("artifacts/robustness_comparison/reference/plantvillage/predictions.csv")
    cases = []
    for index, name in enumerate(["Scab", "Black Rot", "Cedar Rust", "Healthy"]):
        # First by path, regardless of recorded prediction: no favorable-example selection.
        row = pv[pv.label == index].sort_values("path").iloc[0]
        cases.append({"case": name, "path": row.path, "expected": row.class_name.replace("Apple___", "").replace("_", " "),
                      "provenance": "PlantVillage held-out test label"})
    field = pd.read_csv("artifacts/robustness_comparison/reference/kashmir_scab_healthy/predictions.csv")
    difficult = field[(field.label == 0) & (field.predicted_class != field.true_class)]
    row = difficult.sort_values(["confidence", "path"], ascending=[False, True]).iloc[0]
    cases.append({"case": "Difficult field image", "path": row.path, "expected": "Apple scab",
                  "provenance": "Kashmir evaluation-only label, preliminary; selected as a known high-confidence failure"})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/demo_workflow_test"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    paths = [DEFAULT_CLASSIFIER, DEFAULT_SEGMENTER, DEFAULT_COMPARISON]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    pipeline = DemoPipeline(device="cpu")
    assert not pipeline.classifier.training
    assert all(not model.training and not any(p.requires_grad for p in model.parameters())
               for model, _ in pipeline.segmenters.values())
    assert all(threshold == 0.70 for _, threshold in pipeline.segmenters.values())
    captured = {}

    def capture(name):
        def hook(model, inputs):
            captured[name] = inputs[0].detach().cpu().clone()
        return hook

    hooks = [pipeline.classifier.register_forward_pre_hook(capture("classifier"))]
    for name, (model, _) in pipeline.segmenters.items():
        hooks.append(model.register_forward_pre_hook(capture(name)))
    app = build_demo(pipeline).queue(max_size=8)
    results = []
    visuals = []
    try:
        app.launch(server_name="127.0.0.1", server_port=args.port, share=False,
                   inbrowser=False, prevent_thread_lock=True, quiet=True)
        client = Client(f"http://127.0.0.1:{args.port}", verbose=False)
        for case_index, case in enumerate(select_cases()):
            with Image.open(case["path"]) as source:
                original = prepare_image(source)
            for name in pipeline.segmenters:
                captured.clear()
                scores, source_path, overlay_path, mask_path, details = client.predict(
                    handle_file(case["path"]), name, api_name="/analyze",
                )
                actual_classifier = captured["classifier"]
                actual_segmenter = captured[name]
                torch.testing.assert_close(actual_classifier, expected_tensor(original, 224, InterpolationMode.BICUBIC), rtol=0, atol=0)
                torch.testing.assert_close(actual_segmenter, expected_tensor(original, 256, InterpolationMode.BILINEAR), rtol=0, atol=0)
                assert tuple(actual_classifier.shape) == (1, 3, 224, 224)
                assert tuple(actual_segmenter.shape) == (1, 3, 256, 256)
                with torch.inference_mode():
                    probabilities = pipeline.classifier(actual_classifier).softmax(1)[0].numpy()
                    model, threshold = pipeline.segmenters[name]
                    binary = model(actual_segmenter).sigmoid()[0, 0].numpy() >= threshold
                api_scores = {item["label"]: item["confidence"] for item in scores["confidences"]}
                np.testing.assert_allclose([api_scores[label] for label in pipeline.labels], probabilities, rtol=0, atol=1e-6)
                assert abs(sum(api_scores.values()) - 1) < 1e-6
                assert scores["label"] == pipeline.labels[int(probabilities.argmax())]
                with Image.open(source_path) as image:
                    np.testing.assert_array_equal(np.asarray(image), np.asarray(original))
                with Image.open(mask_path) as image:
                    mask = image.convert("L").copy()
                with Image.open(overlay_path) as image:
                    overlay = image.convert("RGB").copy()
                assert overlay.size == mask.size == original.size
                assert set(np.unique(np.asarray(mask))).issubset({0, 255})
                expected_mask = Image.fromarray(binary.astype(np.uint8) * 255).resize(original.size, Image.Resampling.NEAREST)
                np.testing.assert_array_equal(np.asarray(mask), np.asarray(expected_mask))
                selected = np.asarray(mask) > 0
                expected_overlay = np.array(original)
                expected_overlay[selected] = np.rint(0.55 * expected_overlay[selected] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
                np.testing.assert_array_equal(np.asarray(overlay), expected_overlay)
                disagreement = scores["label"] == "healthy" and bool(binary.any())
                assert ("Warning: model disagreement" in details) == disagreement
                record = {**case, "segmenter": name, "predicted": scores["label"],
                          "confidence": max(api_scores.values()), "matches_dataset_label": scores["label"] == case["expected"],
                          "scores": api_scores, "grid_area": float(binary.mean()), "healthy_disagreement_warning": disagreement,
                          "classifier_input_shape": list(actual_classifier.shape), "segmenter_input_shape": list(actual_segmenter.shape),
                          "output_size": list(original.size), "preprocessing_exact_match": True,
                          "mask_and_overlay_exact_match": True, "details": details}
                results.append(record)
                suffix = "reference" if name == PRIMARY else "hardneg05"
                overlay.save(args.output_dir / f"case_{case_index}_{suffix}_overlay.png")
                mask.save(args.output_dir / f"case_{case_index}_{suffix}_mask.png")
                if name == PRIMARY:
                    visuals.append((case, original, overlay, mask, record))
                print(f"{case['case']} | {name} | {scores['label']} ({record['confidence']:.2%}) | workflow PASS", flush=True)
    finally:
        app.close()
        for hook in hooks:
            hook.remove()
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert before == after
    figure, axes = plt.subplots(5, 3, figsize=(12, 18))
    for axes_row, (case, original, overlay, mask, record) in zip(axes, visuals):
        for axis, image in zip(axes_row, [original, overlay, mask]):
            axis.imshow(image, cmap="gray" if image.mode == "L" else None, vmin=0, vmax=255)
            axis.axis("off")
        axes_row[0].set_title(f"{case['case']}: dataset label {case['expected']}", fontsize=10)
        axes_row[1].set_title(f"Prediction: {record['predicted']} ({record['confidence']:.1%})", fontsize=10)
        axes_row[2].set_title("Separate binary segmentation output", fontsize=10)
    figure.tight_layout()
    figure.savefig(args.output_dir / "workflow_contact_sheet.png", dpi=130)
    plt.close(figure)
    report = {"workflow_passed": True, "cases": 5, "inference_paths": len(results),
              "checkpoint_hashes_unchanged": True, "checkpoint_sha256": before,
              "limitations": "Five-case integration check, not an accuracy benchmark. Kashmir labels remain preliminary. Alignment does not establish lesion correctness.",
              "results": results}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Five-case workflow test", "", "All 10 real Gradio upload paths passed integration checks. Checkpoint hashes are unchanged.", "",
             "| Case | Segmenter | Dataset label | Classifier output | Confidence | Mask area | Healthy disagreement |",
             "| --- | --- | --- | --- | ---: | ---: | --- |"]
    for record in results:
        segmenter_label = "Reference" if record["segmenter"] == PRIMARY else "Hardneg05"
        lines.append(f"| {record['case']} | {segmenter_label} | {record['expected']} | {record['predicted']} | {record['confidence']:.2%} | {record['grid_area']:.2%} | {record['healthy_disagreement_warning']} |")
    lines += ["", report["limitations"], "", "Verified exact model-input preprocessing, class-score parity, original pixel preservation, thresholded mask mapping, overlay pixels, saved thresholds, and conditional healthy-disagreement warnings.", "", "![Workflow contact sheet](workflow_contact_sheet.png)"]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
