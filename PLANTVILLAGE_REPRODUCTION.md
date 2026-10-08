# PlantVillage saved-checkpoint reproduction

On 2026-10-08, both frozen classifiers exactly reproduced their original
PlantVillage test scores. No retraining or checkpoint changes were made.

| Model | Correct / total | Expected accuracy | Rerun accuracy | Macro-F1 |
| --- | --- | --- | --- | --- |
| Baseline | 429 / 475 | 90.32% | 90.32% | 0.911271 |
| Background augmented | 398 / 475 | 83.79% | 83.79% | 0.795268 |

For each model, the full ordered label vector, predicted class vector, and all
four probabilities per image are identical to the original `test_predictions.npz`.
There are zero prediction disagreements and zero maximum probability difference.
Accuracy and macro-F1 also exactly match checkpoint metadata.

## Checks

- Used `artifacts/apple_baseline_checkpoint.pt` and
  `artifacts/background_augmented/apple_background_augmented_checkpoint.pt`.
  Their SHA-256 hashes match those recorded in the Kashmir evaluation.
- Class mapping is scab = 0, black rot = 1, cedar apple rust = 2, healthy = 3,
  matching both checkpoints and the evaluation dataset.
- Reconstructed the original split from cached Kaggle dataset version 3:
  sorted paths, decoded RGB exact-image deduplication, then stratified 70/15/15
  splitting with seed 42. Of 3,171 Apple images, 3,164 remain after removing
  seven duplicates: 2,214 train, 475 validation, and 475 test.
- Both reconstructed test manifests are identical. Test support is 95 scab,
  93 black rot, 41 cedar rust, and 246 healthy. Image readability was verified;
  the split routine verified no decoded exact-image overlap between splits.
- Used the shared original preprocessing: PIL RGB conversion, bilinear resize
  to 128 x 128, and ToTensor, without normalization or background replacement.
  Inference ran on CPU in evaluation mode, with dropout disabled.

The root-level `apple_baseline_checkpoint.pt` is an older notebook artifact with
92.84% recorded accuracy. It is separate from the reproducible pipeline baseline
and was not used for this check or yesterday's Kashmir evaluation.

Original training runs did not save per-image split manifests, so historical
image identity cannot be independently compared against a contemporaneous
manifest. Exact reproduction of every saved probability, not just aggregate
accuracy, strongly supports that the original test setup was recovered. This
check does not audit near duplicates or train/test independence beyond exact
decoded-image overlap.

## Reproduction and outputs

Run from the project root with a new output directory:

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe scripts/reproduce_plantvillage.py `
  --dataset-path C:/Users/gunn/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3 `
  --output-dir artifacts/plantvillage_reproduction_2026-10-08
```

The script refuses to overwrite an existing output directory. Each model's
directory contains train/validation/test manifests with decoded-image SHA-256,
metrics JSON, per-image predictions and probabilities, compressed predictions,
and confusion matrices. The summary records runtime versions and checkpoint
hashes. A durable summary is saved in
`results/2026-10-08_plantvillage_reproduction/summary.json`.

## Implication for Kashmir

This reproduction check passes. There is no discrepancy here requiring a
checkpoint, preprocessing, or split repair before interpreting the Kashmir
scores. It confirms that the same checkpoints perform as expected on their
original test distribution; it does not validate Kashmir's labels or establish
why external performance is weak. Verification of missing Kashmir images and
the provisional rot-to-black-rot label mapping remains the next task. Demo
integration remains paused.
