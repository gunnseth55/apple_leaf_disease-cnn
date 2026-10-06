# Healthy structural-negative benchmark

This evaluation-only benchmark contains 50 healthy Apple leaf images from the
Plant Pathology 2020 healthy-labelled pool. None of the images appears in the
lesion training manifest or among the 40 selected hard negatives.

## Integrity and composition

- All 50 source rows have `healthy == 1` in the source labels.
- Every lesion mask is all-zero and is checked after creation.
- SHA-256 found no exact overlap with the 370 lesion-training images or within
  the benchmark.
- A 256-bit difference hash found no training or internal match at Hamming
  distance 4 or less. The nearest observed distance was 91.
- The manifest marks every row `evaluation_only`; it must not be passed to the
  training command or used for hard-negative selection.
- Categories are multi-label: natural background (50), edge (29), sunlight
  (28), shadow (24), fold (23), stem (19), vein (14), edge damage (7), and hole
  (6).

The fixed manifest and validation evidence are in
`data/healthy_structural_benchmark/manifest.csv`, `duplicate_audit.csv`, and
`validation_report.json`. Rebuild them with `scripts/build_healthy_benchmark.py`.

## Frozen-checkpoint evaluation

All checkpoints were evaluated without retraining at their stored threshold of
0.70 on 256x256 inputs.

| Checkpoint | Images with any prediction | FP pixel rate | Mean area (px) | Components | Mean component (px) | Max component (px) | Mean probability | Maximum probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paired augmentation | 100% | 8.1856% | 5,364.52 | 1,821 | 147.30 | 5,001 | 0.2526 | 1.0000 |
| Preferred 30/70 | 98% | 5.3483% | 3,505.06 | 1,754 | 99.92 | 2,675 | 0.3513 | 1.0000 |
| 30/70 + 5% hard negatives | 94% | 0.3578% | 234.48 | 648 | 18.09 | 310 | 0.4119 | 1.0000 |

The hard-negative checkpoint reduces false-positive pixels and mean predicted
area by 93.3% relative to pure 30/70. It reduces component count by 63.1% and
maximum component size by 88.4%. The FP pixel-rate reduction is present in every
category: edge 93.1%, edge damage 94.2%, fold 94.4%, hole 94.2%, shadow 89.5%,
stem 93.4%, sunlight 93.9%, and vein 94.5%.

The image-level any-prediction metric is intentionally strict: a single pixel
counts as a false alarm. It therefore remains high despite the very large drop
in false-positive area and size. Likewise, the higher mean probability of the
5% model alongside lower thresholded area indicates different score
calibration; probability statistics should not be read as area metrics.

Detailed outputs are under `artifacts/healthy_structural_benchmark_evaluation/`:

- `checkpoint_summary.csv`
- `category_summary.csv`
- one per-image CSV per checkpoint
- `results.json`

## Decision

The 5% model establishes a real and substantial healthy-structure benefit while
giving up 0.0033 pooled Dice on diseased test images. This supports investigating
a controlled dual-objective or separate healthy-batch loss. Do not increase the
hard-negative share blindly, and retain the pure 30/70 checkpoint as the general
reference until that controlled experiment is evaluated on both test sets.
