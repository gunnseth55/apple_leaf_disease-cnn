# C1 held-out three-model comparison, 2026-10-22

Full procedure and interpretation:
[C1_HELDOUT_EVALUATION.md](../../C1_HELDOUT_EVALUATION.md).

| Model | PV accuracy | PV macro-F1 | PP2020 accuracy | PP2020 macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Reference | 99.79% (474/475) | 0.9982 | 48.41% (122/252) | 0.4347 |
| Robust | 100.00% (475/475) | 1.0000 | 56.75% (143/252) | 0.5528 |
| C1 | 99.79% (474/475) | 0.9982 | 94.84% (239/252) | 0.9465 |

Each model used the same ordered test images. PV macro-F1 uses four classes;
PP2020 macro-F1 uses its three supported classes, retaining all four output
classes and counting black-rot predictions as errors. PP2020 black-rot recall
is unmeasured. Sources are reported separately. No training or tuning occurred.

- [comparison.csv](comparison.csv): full metrics, recall, confidence diagnostics.
- [paired_comparison.csv](paired_comparison.csv): corrected/regressed counts.
- [input_validation.json](input_validation.json): frozen provenance and audit.
- [verification_report.json](verification_report.json): independent checks of
  all six saved evaluations and reproduction of original reference/robust PV
  predictions.
- `reference/`, `robust/`, `c1/`: per-source predictions, errors, metrics,
  confusion matrices, confidence bins and figures.
- `*_test_manifest.csv` and `*_paired_predictions.csv`: exact image identities,
  order and aligned model outputs.
- `evaluation_script.py` and `verification_script.py`: exact scripts at the time
  of this run. The canonical maintained versions live under `src/` and `scripts/`.
- `snapshot_sha256.json`: hashes of the snapshot files, excluding the hash file
  itself.

Original local run: `artifacts/c1_heldout_comparison_v1/`. No weights or source
images are included here. The original absolute image paths are provenance;
reproduction requires those local inputs or an explicitly verified relocation.
Input-validation rechecks require the original frozen files, while saved CSV/NPZ
predictions and figures can be inspected without them.

Results support improved recognition on held-out PP2020 with reference PV
accuracy/macro-F1 preserved. They do not establish new-orchard generalization,
field black-rot recall, calibration, or segmentation accuracy. Tests have now
been observed and must not guide subsequent tuning. The demo was not changed.
