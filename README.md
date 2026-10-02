# Apple Leaf Disease Classifier

This project trains and evaluates the four-class baseline CNN that was originally
developed in `apple_leaf_baseline.ipynb`. The notebook remains available for
exploration, while the Python pipeline runs in a reproducible order without
depending on notebook state.

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

Each command also works without installing console entry points:

```powershell
$env:PYTHONPATH = "src"
python -m appleleaf.cli.train
```
