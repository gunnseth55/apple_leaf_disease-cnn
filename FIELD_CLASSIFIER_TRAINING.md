# C1 mixed-source EfficientNet-B0 fine-tuning

Run from the project root:

```powershell
.venv\Scripts\python.exe -m appleleaf.cli.train_field --epochs 10 --batch-size 16 --device auto
```

Optional data/checkpoint checks without optimization:

```powershell
.venv\Scripts\python.exe -m appleleaf.cli.train_field --dry-run
```

The dry run was verified locally. It performs no epochs, optimizer steps, model
inference, checkpoint writes or training-output creation. The full optimization
loop is deliberately left for the user to run.

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
