"""Recheck both frozen segmenters on identical saved crops, without filtering masks."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from appleleaf.demo import DEFAULT_SEGMENTER, DEFAULT_COMPARISON, lesion_overlay, segmentation_input
from appleleaf_lesion.workflow import load_model


def main():
    torch.set_num_threads(4)
    source = Path("artifacts/demo_crop_test")
    destination = Path("artifacts/demo_segmenter_recheck")
    destination.mkdir(parents=True, exist_ok=True)
    prior = json.loads((source / "report.json").read_text(encoding="utf-8"))
    cases = prior["results"][::2]
    paths = {"30/70": DEFAULT_SEGMENTER, "Hardneg05": DEFAULT_COMPARISON}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    models = {name: load_model(path, torch.device("cpu")) for name, path in paths.items()}
    for model, metadata in models.values():
        model.eval().requires_grad_(False)
        assert metadata["image_size"] == 256
        assert metadata["threshold"] == 0.70
    # Visually checked background rectangles in the two saved user-upload crops.
    # These are evaluation regions only; they never modify inputs or predicted masks.
    regions = {5: [(550, 0, 615, 70), (580, 270, 615, 350)],
               6: [(0, 355, 95, 412)]}
    results = []
    for index, case in enumerate(cases):
        with Image.open(source / f"case_{index}_crop.png") as opened:
            image = opened.convert("RGB")
        tensor = segmentation_input(image).unsqueeze(0)
        roi = np.zeros((image.height, image.width), dtype=bool)
        for left, top, right, bottom in regions.get(index, []):
            assert 0 <= left < right <= image.width and 0 <= top < bottom <= image.height
            roi[top:bottom, left:right] = True
        panels = []
        row = {"case": case["case"], "crop_box": case["crop_box"], "crop_size": image.size,
               "crop_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
               "background_roi_boxes": regions.get(index, []), "background_roi_pixels": int(roi.sum()),
               "models": {}}
        for name, (model, metadata) in models.items():
            with torch.inference_mode():
                probability = model(tensor).sigmoid()[0, 0].numpy()
            threshold = float(metadata["threshold"])
            binary = probability >= threshold
            overlay, mask = lesion_overlay(image, probability, threshold)
            mask_array = np.asarray(mask) > 0
            # Confirm identical outputs to the earlier crop comparison for the same input.
            suffix = "reference" if name == "30/70" else "hardneg05"
            with Image.open(source / f"case_{index}_{suffix}_mask.png") as previous:
                np.testing.assert_array_equal(np.asarray(mask), np.asarray(previous.convert("L")))
            mask.save(destination / f"case_{index}_{suffix}_mask.png")
            overlay.save(destination / f"case_{index}_{suffix}_overlay.png")
            np.save(destination / f"case_{index}_{suffix}_probability.npy", probability)
            bg_count = int((mask_array & roi).sum()) if roi.any() else None
            row["models"][name] = {"threshold": threshold, "input_shape": list(tensor.shape),
                                   "predicted_grid_pixels": int(binary.sum()), "predicted_grid_fraction": float(binary.mean()),
                                   "background_roi_selected_pixels": bg_count,
                                   "background_roi_selected_fraction": bg_count / int(roi.sum()) if roi.any() else None,
                                   "matches_previous_mask_exactly": True}
            panels.append((name, probability, mask, overlay, binary))
        a, b = panels[0][4], panels[1][4]
        union = int((a | b).sum())
        row["mask_overlap_iou"] = float((a & b).sum() / union) if union else 1.0
        results.append(row)
        figure, axes = plt.subplots(2, 4, figsize=(14, 8))
        for axes_row, (name, probability, mask, overlay, _) in zip(axes, panels):
            for axis, panel, title in zip(axes_row, [image, probability, mask, overlay],
                                         ["Same selected crop", f"{name}: sigmoid probabilities", "Binary mask at saved 0.70", f"{name}: 45% red overlay"]):
                axis.imshow(panel, cmap="magma" if isinstance(panel, np.ndarray) else "gray" if panel.mode == "L" else None,
                            vmin=0, vmax=1 if isinstance(panel, np.ndarray) else 255)
                axis.set_title(title, fontsize=10)
                axis.axis("off")
        figure.suptitle(case["case"] + " — raw predictions, no smoothing or region removal")
        figure.tight_layout()
        figure.savefig(destination / f"case_{index}_comparison.png", dpi=120)
        plt.close(figure)
        print(f"{case['case']}: 30/70={a.mean():.2%}, Hardneg05={b.mean():.2%}; background ROI counts "
              f"{row['models']['30/70']['background_roi_selected_pixels']} / {row['models']['Hardneg05']['background_roi_selected_pixels']}", flush=True)
    assert hashes == {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    report = {"same_crop_and_tensor_for_both_models": True, "checkpoint_hashes_unchanged": True,
              "checkpoint_sha256": hashes, "results": results,
              "limitations": "Mask overlap measures agreement, not accuracy. Total selected area is not false-positive area. Background counts cover only specified visually checked rectangles, not a complete leaf mask. User-upload disease labels and lesion boundaries are unverified."}
    (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Frozen segmenter recheck", "", "Both models received the same saved crop and identical normalized 256 × 256 tensor. Both retained their saved 0.70 thresholds. All masks exactly reproduced the previous paired crop run; hashes are unchanged.", "",
             "| Case | 30/70 selected grid area | Hardneg05 selected grid area | Mask agreement IoU |",
             "| --- | ---: | ---: | ---: |"]
    for row in results:
        lines.append(f"| {row['case']} | {row['models']['30/70']['predicted_grid_fraction']:.2%} | {row['models']['Hardneg05']['predicted_grid_fraction']:.2%} | {row['mask_overlap_iou']:.3f} |")
    lines += ["", "## Visible background checks", "", "| Case | Reviewed background pixels | 30/70 selected pixels | Hardneg05 selected pixels |", "| --- | ---: | ---: | ---: |"]
    for row in results:
        if row["background_roi_pixels"]:
            lines.append(f"| {row['case']} | {row['background_roi_pixels']} | {row['models']['30/70']['background_roi_selected_pixels']} | {row['models']['Hardneg05']['background_roi_selected_pixels']} |")
    lines += ["", report["limitations"], "", "All probability maps, binary masks and overlays are saved beside this report. No background rectangles were used to suppress predictions.", "",
              "![First user-upload crop comparison](case_5_comparison.png)", "", "![Second user-upload crop comparison](case_6_comparison.png)"]
    (destination / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
