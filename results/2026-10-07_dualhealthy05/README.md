# Dualhealthy05 experiment result snapshot

Recorded 2026-10-07 after the completed 30-epoch controlled experiment.
This directory preserves the existing local CSV/JSON outputs in Git; recording
did not retrain or rerun either benchmark. See ../../PROJECT_PROGRESS.md for
the question, controls, interpretation, limitations, and decision.

## Stored evidence

- `checkpoint_metadata.json`: metadata read from the saved checkpoint, excluding
  weights. Restored epoch 18, threshold 0.55, test loss 0.4344685475031535.
- `lesion_detection_real30_synthetic70_dualhealthy05/training_history.csv`:
  all 30 epochs, including auxiliary loss and mean training-healthy probability.
- `lesion_evaluation_real30_synthetic70_dualhealthy05/`: pooled JSON metrics
  and full per-disease and per-image CSV metrics for the 30 diseased test images.
- `healthy_structural_benchmark_evaluation_dualhealthy05/`: full four-model
  checkpoint summary, category summary, per-image CSVs, and JSON results on
  50 held-out healthy images.

Validation Dice at epoch 18 was 0.685645966177423 before calibration at 0.50.
The terminal output reported validation Dice 0.6878 after calibration to 0.55;
the checkpoint stores the threshold but not the calibrated validation metrics.
Test Dice is 0.6933464076324352. Mean image Dice is 0.702884442598748.

The healthy-loss weight 0.05 means a separate weighted BCE objective, not 5%
sampling. The training healthy loader selects the 40 hard-negative rows of
`train.csv`; the 50 benchmark images are evaluation-only. Checkpoints retain
their own validation-calibrated thresholds: 0.70 for the first three models
and 0.55 for dualhealthy05. No threshold was selected using test results.

Checkpoint weights and the 30 overlay PNGs remain in the ignored local artifact
directories. This snapshot does not contain those files or the source images.
CSV/JSON content is preserved with normalized line endings and a final newline.

## Source identity (SHA-256)

Hashes below identify original files at recording time, before line-ending
normalization. Paths are relative to the repository root.

| File | SHA-256 |
| --- | --- |
| `artifacts/lesion_detection_real30_synthetic70_dualhealthy05/unet_resnet34_lesion_checkpoint.pt` | `1480E5919A23A7D891A01905D9D4BCEF44A77781C6A62F7BA62B1185D424AB82` |
| `lesion_manifests/train_before_hard_negatives.csv` | `73849BFF9EC34E54BCE852805556BA3F118D0D4A45F83B3E6E097F80539CEDA8` |
| `lesion_manifests/train.csv` | `98EABA54793D25DDD3C2B12B114C98136410B6C5A244C0DFBCBD2BB1FC042436` |
| `lesion_manifests/val.csv` | `7C3B1C892D089B11535339798F9EF6872E5DE5C734AF84BF567D16DCA2CA04BB` |
| `lesion_manifests/test.csv` | `E77428D812FBD8FA3D328101B095A3053624858D5AFE68405D73D2023147542C` |
| `data/healthy_structural_benchmark/manifest.csv` | `E4CDA5572CFD47A4A5FB100BDAA256E874F5C6A75CD3D7C42A636DACF79E1BAD` |

## Reproduction commands

These document the completed workflow. Run from the repository root with the
project environment active and source data present. The benchmark is paused
for now; these commands are retained for reproducibility.

```powershell
python -m appleleaf_lesion.cli.train `
  --manifest-dir lesion_manifests `
  --train-manifest lesion_manifests\train_before_hard_negatives.csv `
  --real-sampling-fraction 0.30 `
  --samples-per-epoch 330 `
  --healthy-manifest lesion_manifests\train.csv `
  --healthy-loss-weight 0.05 `
  --epochs 30 `
  --batch-size 8 `
  --image-size 256 `
  --artifacts-dir artifacts\lesion_detection_real30_synthetic70_dualhealthy05 `
  --device auto

python -m appleleaf_lesion.cli.evaluate `
  --checkpoint artifacts\lesion_detection_real30_synthetic70_dualhealthy05\unet_resnet34_lesion_checkpoint.pt `
  --manifest lesion_manifests\test.csv `
  --output-dir artifacts\lesion_evaluation_real30_synthetic70_dualhealthy05 `
  --examples 30 `
  --device auto

python scripts\evaluate_healthy_benchmark.py `
  --manifest data\healthy_structural_benchmark\manifest.csv `
  --checkpoint artifacts\lesion_detection\unet_resnet34_augmented_before_hard_negatives.pt `
  --name paired_augmentation `
  --checkpoint artifacts\lesion_detection_real30_synthetic70\unet_resnet34_lesion_checkpoint.pt `
  --name real30_synthetic70 `
  --checkpoint artifacts\lesion_detection_real30_synthetic70_hardneg05\unet_resnet34_lesion_checkpoint.pt `
  --name real30_synthetic70_hardneg05 `
  --checkpoint artifacts\lesion_detection_real30_synthetic70_dualhealthy05\unet_resnet34_lesion_checkpoint.pt `
  --name real30_synthetic70_dualhealthy05 `
  --output-dir artifacts\healthy_structural_benchmark_evaluation_dualhealthy05 `
  --device auto
```

## Decision and verification

Keep pure 30/70 as the general lesion reference and hardneg05 as the healthy
FP-area reference. Dualhealthy05 has lower diseased Dice than both and higher
healthy FP area than hardneg05 at their stored operating points. Preserve it
as experimental evidence and pause healthy benchmark checking for now.

All six existing lesion unit tests passed with
`.venv/Scripts/python.exe -m unittest discover -s tests -v`.
