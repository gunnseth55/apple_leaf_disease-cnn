"""Paired full-image/crop API checks with fixed, recorded manual crop boxes."""

import hashlib
import json
from pathlib import Path
import tempfile

from gradio_client import Client, handle_file
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode

from appleleaf.demo import DEFAULT_CLASSIFIER, DEFAULT_SEGMENTER, DEFAULT_COMPARISON, DemoPipeline, PRIMARY, build_demo, prepare_image
from test_demo_workflow import select_cases, expected_tensor


def main():
    torch.set_num_threads(4)
    folder = Path("artifacts/demo_crop_test")
    folder.mkdir(parents=True, exist_ok=True)
    paths = [DEFAULT_CLASSIFIER, DEFAULT_SEGMENTER, DEFAULT_COMPARISON]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    pipeline = DemoPipeline(device="cpu")
    cases = select_cases()
    boxes = [(20, 20, 245, 250), (15, 15, 242, 244), (20, 15, 245, 245), (10, 8, 246, 250), (20, 10, 302, 780)]
    for case, box in zip(cases, boxes):
        case["crop_box"] = box
    # Previously inspected leaf screenshots; unknown disease labels are not inferred.
    for filename, box in [("Screenshot 2026-10-13 154217.png", (50, 5, 665, 355)),
                          ("Screenshot 2026-10-13 154344.png", (0, 20, 400, 432))]:
        matches = list(Path(tempfile.gettempdir(), "gradio").glob(f"*/{filename}"))
        if matches:
            cases.append({"case": filename, "path": str(matches[0]), "expected": None,
                          "provenance": "Recent user upload; disease label unverified", "crop_box": box})
    captured = {}
    def capture(name):
        def hook(model, inputs):
            captured.setdefault(name, []).append(inputs[0].detach().cpu().clone())
        return hook
    hooks = [pipeline.classifier.register_forward_pre_hook(capture("classifier"))]
    for name, (model, _) in pipeline.segmenters.items():
        hooks.append(model.register_forward_pre_hook(capture(name)))
    app = build_demo(pipeline).queue()
    records, visuals = [], []
    try:
        app.launch(server_name="127.0.0.1", server_port=7862, share=False, inbrowser=False,
                   prevent_thread_lock=True, quiet=True)
        client = Client("http://127.0.0.1:7862", verbose=False)
        for index, case in enumerate(cases):
            with Image.open(case["path"]) as source:
                full = prepare_image(source)
            left, top, right, bottom = case["crop_box"]
            assert 0 <= left < right <= full.width and 0 <= top < bottom <= full.height
            crop = full.crop(case["crop_box"])
            crop_path = folder / f"case_{index}_crop.png"
            crop.save(crop_path)
            # Exercise the same upload-to-editor initialization used by the UI.
            seeded = client.predict(handle_file(case["path"]), api_name="/seed_crop")
            assert seeded["composite"] is not None
            with Image.open(seeded["composite"]) as seeded_image:
                np.testing.assert_array_equal(np.asarray(seeded_image), np.asarray(full))
            editor = {"background": handle_file(case["path"]), "layers": [], "composite": handle_file(str(crop_path))}
            for name, (model, threshold) in pipeline.segmenters.items():
                captured.clear()
                result = client.predict(handle_file(case["path"]), editor, name, api_name="/compare_crop")
                classifier_inputs = list(captured["classifier"])
                segmenter_inputs = list(captured[name])
                assert len(classifier_inputs) == len(segmenter_inputs) == 2
                for image, cinputs, sinputs in zip([full, crop], classifier_inputs, segmenter_inputs):
                    torch.testing.assert_close(cinputs, expected_tensor(image, 224, InterpolationMode.BICUBIC), rtol=0, atol=0)
                    torch.testing.assert_close(sinputs, expected_tensor(image, 256, InterpolationMode.BILINEAR), rtol=0, atol=0)
                with torch.inference_mode():
                    full_probs = pipeline.classifier(classifier_inputs[0]).softmax(1)[0].numpy()
                    crop_probs = pipeline.classifier(classifier_inputs[1]).softmax(1)[0].numpy()
                    binary = model(segmenter_inputs[1]).sigmoid()[0, 0].numpy() >= threshold
                api = {item["label"]: item["confidence"] for item in result[0]["confidences"]}
                np.testing.assert_allclose([api[label] for label in pipeline.labels], crop_probs, atol=1e-6, rtol=0)
                with Image.open(result[1]) as returned:
                    np.testing.assert_array_equal(np.asarray(returned), np.asarray(crop))
                with Image.open(result[3]) as returned:
                    mask = returned.convert("L").copy()
                expected_mask = Image.fromarray(binary.astype(np.uint8)*255).resize(crop.size, Image.Resampling.NEAREST)
                np.testing.assert_array_equal(np.asarray(mask), np.asarray(expected_mask))
                with Image.open(result[2]) as returned:
                    overlay = returned.convert("RGB").copy()
                selected = np.asarray(mask) > 0
                expected_overlay = np.array(crop)
                expected_overlay[selected] = np.rint(.55*expected_overlay[selected]+.45*np.array([255,0,0])).astype(np.uint8)
                np.testing.assert_array_equal(np.asarray(overlay), expected_overlay)
                assert "Softmax score:" in result[4] and "Softmax confidence" not in result[4]
                assert "Full-image softmax score" in result[5]
                cropped_label = pipeline.labels[int(crop_probs.argmax())]
                disagreement = cropped_label == "healthy" and bool(binary.any())
                assert ("Warning: model disagreement" in result[4]) == disagreement
                record = {**case, "segmenter": name, "full_size": full.size, "crop_size": crop.size,
                          "full_predicted": pipeline.labels[int(full_probs.argmax())], "crop_predicted": cropped_label,
                          "full_score": float(full_probs.max()), "crop_score": float(crop_probs.max()),
                          "full_scores": dict(zip(pipeline.labels, map(float, full_probs))), "crop_scores": api,
                          "crop_grid_area": float(binary.mean()), "preprocessing_and_alignment_passed": True,
                          "healthy_disagreement_warning": disagreement}
                records.append(record)
                suffix = "reference" if name == PRIMARY else "hardneg05"
                overlay.save(folder / f"case_{index}_{suffix}_overlay.png")
                mask.save(folder / f"case_{index}_{suffix}_mask.png")
                if name == PRIMARY:
                    visuals.append((case, full, crop, mask, overlay, record))
                print(f"{case['case']} | {suffix}: {record['full_predicted']} -> {cropped_label} | PASS", flush=True)
    finally:
        app.close()
        for hook in hooks:
            hook.remove()
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = {"cases": len(cases), "paired_paths": len(records), "checkpoint_hashes_unchanged": True,
              "checkpoint_sha256": hashes, "crop_selection": "Fixed manual rectangles based on leaf location, not model scores; no crop optimization or training.",
              "results": records}
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Crop comparison", "", "Both models received the identical crop with their own original training preprocessing. All paired API checks passed; masks retain every threshold-selected region.", "",
             "| Case | Full prediction | Full softmax score | Crop prediction | Crop softmax score |",
             "| --- | --- | ---: | --- | ---: |"]
    for r in records:
        if r["segmenter"] == PRIMARY:
            lines.append(f"| {r['case']} | {r['full_predicted']} | {r['full_score']:.2%} | {r['crop_predicted']} | {r['crop_score']:.2%} |")
    lines += ["", report["crop_selection"], "", "A higher or changed softmax score does not prove improved accuracy. Kashmir labels remain preliminary and the recent uploads have no verified disease labels.", "", "![Full image, crop, binary mask, overlay](contact_sheet.png)"]
    (folder / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    figure, axes = plt.subplots(len(visuals), 4, figsize=(15, 4*len(visuals)))
    for row, (case, full, crop, mask, overlay, record) in zip(axes, visuals):
        for axis, image, title in zip(row, [full, crop, mask, overlay],
                                     [f"Full: {record['full_predicted']}", f"Crop: {record['crop_predicted']}", "Binary mask", "45% red overlay"]):
            axis.imshow(image, cmap="gray" if image.mode == "L" else None)
            axis.set_title(title)
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(folder / "contact_sheet.png", dpi=100)
    plt.close(figure)
    print(f"Report: {folder / 'REPORT.md'}")


if __name__ == "__main__":
    main()
