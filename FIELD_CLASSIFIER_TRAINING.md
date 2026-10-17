# C1 mixed-source EfficientNet-B0 fine-tuning

## Completed user-run experiment: 2026-10-18

All 10 epochs completed. **Epoch 10** achieved the highest validation selection
score, **0.975387500653458**, and was saved as the selected checkpoint. These are
validation results; neither test set has been evaluated for this run.

| Validation source | Images | Accuracy | Macro-F1 used for selection |
| --- | ---: | ---: | ---: |
| PlantVillage | 475 | 1.000000 (475/475) | 1.000000 (four classes) |
| PP2020 field | 251 | 0.952191 (239/251) | 0.950775 (three present classes) |

PP2020 all-four-class macro-F1 was 0.713081; its absent black-rot class contributes
zero to that alternative average. Selection uses the equal mean of PV four-class
macro-F1 and PP2020 present-three-class macro-F1.

| Epoch | PV validation macro-F1 | PP2020 validation macro-F1 | Selection score | Epoch seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9963 | 0.7807 | 0.8885 | 880.8 |
| 2 | 1.0000 | 0.8791 | 0.9396 | 782.0 |
| 3 | 1.0000 | 0.8987 | 0.9493 | 847.6 |
| 4 | 1.0000 | 0.9269 | 0.9634 | 759.7 |
| 5 | 0.9982 | 0.9464 | 0.9723 | 772.8 |
| 6 | 1.0000 | 0.9463 | 0.9732 | 744.8 |
| 7 | 1.0000 | 0.9463 | 0.9732 | 736.4 |
| 8 | 1.0000 | 0.9425 | 0.9713 | 754.4 |
| 9 | 1.0000 | 0.9462 | 0.9731 | 908.8 |
| **10** | **1.0000** | **0.9508** | **0.9754** | **759.8** |

Logged epoch time totaled 7,947.0 seconds (about 2 h 12 min), excluding setup and
final serialization. Field validation improved strongly through epoch 5, then
largely plateaued with a small final improvement. Ten epochs was an initial
budget, not a proven optimum; no early stopping was used. These curves alone
do not establish field-test accuracy or justify additional training.

Selected checkpoint:
`artifacts/efficientnet_b0_field_c1_v1/apple_efficientnet_b0_field_c1_checkpoint.pt`.
SHA-256: `2d2b223e25f0d2fc21bf26a3ff158dbc61294043e80a0ca0bac8618ba2a2fac7`.

Verified against `training_history.csv`, `training_summary.json`, `run_config.json`
and selected checkpoint metadata. The original reference hash and all recorded
frozen manifest/report hashes still match. No deployment promotion or further
training occurred while documenting these results.

Next: compare the selected checkpoint with the reference on the same frozen PV
and PP2020 test splits, reporting sources separately. Test results must not guide
tuning. PP2020 still has no orchard black-rot supervision; this run does not
establish field black-rot reliability.

## Reproduction command

Run from the project root:

```powershell
.venv\Scripts\python.exe -m appleleaf.cli.train_field --epochs 10 --batch-size 16 --device auto
```

Optional data/checkpoint checks without optimization:

```powershell
.venv\Scripts\python.exe -m appleleaf.cli.train_field --dry-run
```

The dry run was verified before training. It performs no epochs, optimizer steps,
model inference, checkpoint writes or training-output creation. The user then
executed the 10-epoch run above. The v1 output directory now exists; the trainer
will refuse to overwrite it, including when invoked with `--dry-run`.

## Fixed initial protocol

- Initialize all EfficientNet-B0 parameters from
  `artifacts/efficientnet_b0/apple_efficientnet_b0_checkpoint.pt`; do not reinitialize
  the head, download weights or modify the reference checkpoint.
- Training sources: frozen PlantVillage train (2,214) and PP2020 split-v1 train
  (1,176). The PP2020 builder's full `--verify` runs before loading training data.
- Replacement sampling: 3,390 draws/epoch, expected 50% mass per source and equal
  mass per available class within each source. PV has four classes; PP2020 has
  three. Therefore black rot has 12.5% overall sampling mass, the other classes
  about 29.17% each. Unweighted cross entropy avoids double class weighting.
- Full-model fine-tuning for 10 epochs, AdamW, backbone LR 1e-5, head LR 1e-4,
  weight decay 1e-4 and cosine learning-rate decay; seed 42, batch size 16.
- Full-image horizontal/vertical flips and mild color jitter in training only;
  224px bicubic resize and ImageNet normalization. No random crops. Validation
  preprocessing is deterministic. EXIF orientation is normalized before RGB input.
- Separate validation: all 475 PV and 251 PP2020 validation images. Checkpoint
  score is the mean of PV four-class macro-F1 and PP2020 present-three-class
  macro-F1. Earliest epoch wins ties. All-four-class field macro-F1 is also logged
  so the absent black-rot class is not hidden in reporting.
- No test loaders or test inference. Test manifests are read for provenance and
  duplicate protection only. Final test evaluation is a separate explicit step;
  test performance must not guide checkpoint selection.

Existing paths and recorded pixel identities are checked across both sources'
train/validation/test manifests; overlapping image identities stop the run.
Input manifest and reference checkpoint hashes are checked throughout training.

## Outputs

New directory: `artifacts/efficientnet_b0_field_c1_v1/`.

- `apple_efficientnet_b0_field_c1_checkpoint.pt`: best checkpoint by validation.
- `run_config.json`: initialization hash, sampling, metrics and manifest hashes.
- `train_manifest.csv` and separate source validation manifests.
- `training_history.csv`: source-wise validation scores and epoch timings.
- `training_summary.json`: final selection and preserved-input checks.

Existing run directories are rejected; there is no silent overwrite or automatic
deployment promotion. Use `--artifacts-dir artifacts/efficientnet_b0_field_c1_v2`
only for an explicitly new experiment. `--device auto` uses CUDA when available,
otherwise CPU. The machine currently has a CPU-only PyTorch installation.

PP2020 has no black-rot field category. This experiment retains PlantVillage
black-rot supervision; it does not establish orchard black-rot competence.
