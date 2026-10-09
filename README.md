# Apple Leaf Disease Classifier

This project trains and evaluates the four-class baseline CNN that was originally
developed in `apple_leaf_baseline.ipynb`. The notebook remains available for
exploration, while the Python pipeline runs in a reproducible order without
depending on notebook state.

Two versions of the same CNN are compared:

- **Baseline:** trained on the original PlantVillage colour images.
- **Background augmented:** trained with the same architecture and
  hyperparameters, but with randomized training backgrounds so background
  appearance is unreliable as a class cue.

See [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) for the dated experiment log,
analysis, decisions, and changes made during development.

The user-run EfficientNet-B0 transfer-learning experiment reached **99.79% test
accuracy (474/475)** and **0.9982 macro-F1** on the PlantVillage split. External
evaluation is pending. See [EFFICIENTNET_TRANSFER.md](EFFICIENTNET_TRANSFER.md)
for verified results, the command and protocol.

The external Kashmir classifier evaluation, dataset audit, full metrics, and
reproduction commands are in [KASHMIR_EVALUATION.md](KASHMIR_EVALUATION.md).
Both frozen classifiers achieved 24.23% accuracy on the available 260-image
subset; this does not validate reliable external disease identification.

Both checkpoints exactly reproduced their original PlantVillage test predictions
on 2026-10-08: see [PLANTVILLAGE_REPRODUCTION.md](PLANTVILLAGE_REPRODUCTION.md).

All 47 missing Kashmir images have since been recovered. The
[Kashmir label audit](KASHMIR_LABEL_AUDIT.md) found further conflicting labels;
rot-to-black-rot remains unconfirmed, so the external scores remain preliminary.

Pixel-level lesion segmentation is developed as a separate package and workflow.
See [LESION_DETECTION.md](LESION_DETECTION.md); it does not reuse the
classification training scripts or treat whole-leaf masks as lesion labels.

## Current results

The baseline reached 90.32% test accuracy, while the background-augmented model
reached 83.79%. The controlled validation audit shows why the second model is
still useful:

| Audit condition | Baseline accuracy | Augmented accuracy |
| --- | ---: | ---: |
| Original | 89.05% | 84.63% |
| White background | 46.95% | 80.84% |
| Blue background | 19.79% | 81.26% |
| Black background | 19.79% | 80.42% |
| Background only | 63.16% | 51.79% |

The augmented model sacrifices some original-image accuracy but is substantially
less sensitive to controlled background changes. Its background-only accuracy
falls to the 51.79% majority-class baseline, supporting the conclusion that the
original training data encouraged shortcut learning. These solid-colour tests
measure robustness under controlled shifts; they are not estimates of field
deployment accuracy.

## Setup

Create and activate a virtual environment, then install the project:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch installation can vary by CPU/GPU platform. If you need CUDA-specific
wheels, install PyTorch using the command recommended at https://pytorch.org/
before running `pip install -e .`.

## Run the pipeline

Download the PlantVillage dataset into the KaggleHub cache:

```powershell
appleleaf-download
```

Train and evaluate the baseline model. When `--dataset-path` is omitted, the
dataset is downloaded or reused from the KaggleHub cache automatically:

```powershell
appleleaf-train
```

The command writes the checkpoint, metrics, predictions, training curves, and
confusion matrix to `artifacts/`. To use an existing local dataset or CUDA:

```powershell
appleleaf-train --dataset-path C:\path\to\plantvillage --device cuda
```

Train the same CNN with randomized training backgrounds while leaving validation
and test images unchanged:

```powershell
appleleaf-train --randomize-backgrounds --dataset-path C:\path\to\plantvillage
```

This writes a separate run to `artifacts/background_augmented/`. Twenty-five
percent of training samples retain their original background; the remainder get
a newly sampled stress-test colour (black, white, or blue), arbitrary solid
colour, or smooth random texture each epoch.

Evaluate an existing checkpoint without retraining:

```powershell
appleleaf-evaluate --checkpoint artifacts\apple_baseline_checkpoint.pt
```

Evaluate the augmented checkpoint:

```powershell
appleleaf-evaluate --checkpoint artifacts\background_augmented\apple_background_augmented_checkpoint.pt
```

Run the optional segmented-background sensitivity analysis:

```powershell
appleleaf-background-audit --checkpoint artifacts\apple_baseline_checkpoint.pt
```

The audit evaluates the original images, leaf-preserving solid-background
variants, and a `background_only` condition that masks the leaf while retaining
the original scene. Accuracy above the reported majority-class baseline in the
background-only condition is evidence that class information is leaking through
the background. Solid-colour conditions are stronger distribution-shift stress
tests and should not be interpreted as deployment accuracy.

## Explain individual predictions

Generate Grad-CAM explanations for selected images from the held-out test split:

```powershell
appleleaf-explain `
  --checkpoint artifacts\apple_baseline_checkpoint.pt `
  --dataset-path C:\path\to\plantvillage `
  --output-dir artifacts\explanations\baseline `
  --indices 0 1 2 3 4
```

Omit `--indices` to select two images from every class automatically, avoiding a
healthy-class-heavy sample:

```powershell
appleleaf-explain `
  --checkpoint artifacts\apple_baseline_checkpoint.pt `
  --dataset-path C:\path\to\plantvillage `
  --output-dir artifacts\explanations\balanced\baseline `
  --samples-per-class 2
```

For the background-augmented checkpoint:

```powershell
appleleaf-explain `
  --checkpoint artifacts\background_augmented\apple_background_augmented_checkpoint.pt `
  --dataset-path C:\path\to\plantvillage `
  --output-dir artifacts\explanations\background_augmented `
  --indices 0 1 2 3 4
```

Each output figure contains the exact source image and leaf-mask boundary, the
Grad-CAM influence heatmap, the true and predicted classes, confidence, and all
four class probabilities. `explanations.csv` records these values plus the
fraction of heatmap intensity inside and outside the leaf.

Grad-CAM is a coarse explanation of which regions most influenced a class
score. It does not identify medically validated lesion boundaries and should not
be described as pixel-level disease segmentation.

To see the actual background replacements used during augmented training:

```powershell
appleleaf-preview-augmentation `
  --dataset-path C:\path\to\plantvillage `
  --output artifacts\augmentation_preview.png
```

The current PlantVillage `segmented` directory contains leaf/background masks,
not disease-lesion masks. Exact spot-level detection therefore requires a
pixel-annotated lesion dataset and a separately evaluated segmentation model.

Each command also works without installing console entry points:

```powershell
$env:PYTHONPATH = "src"
python -m appleleaf.cli.train
```
