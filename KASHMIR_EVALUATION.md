# Kashmir external classification evaluation

**Update, 2026-10-08:** All 47 missing images have been recovered into a separate
dataset copy. A visual audit found additional rot/scab label conflicts. The
rot-to-black-rot mapping remains unconfirmed; the scores below are historical
results for the original 260-image subset. See
[KASHMIR_LABEL_AUDIT.md](KASHMIR_LABEL_AUDIT.md) for recovery verification,
all-class review, and the draft 283-image inventory with unverified labels.

Evaluated both frozen classifiers on 2026-10-08, with checkpoint preprocessing
(RGB, resize to 128 × 128, ToTensor) and four-way argmax. No retraining,
threshold selection, cropping, or probability renormalization was performed.

## Dataset audit

The local dataset contains 313 image-named files. Forty-seven contain Git LFS
pointer text rather than images (5 rot, 42 scab). Three byte-identical images
occur in both rot and healthy folders; all six conflicting rows were excluded.
The source files remain unchanged. The evaluated subset contains **260 images**:
117 scab, 100 rot, and 43 healthy. There are no remaining byte-exact duplicates
within this subset. Training-set overlap and near duplicates have not been audited.

Folder labels map as follows:

| Folder | Model class | Status |
| --- | --- | --- |
| SCAB LEAVES | Apple scab | Folder-derived label |
| APPLE ROT LEAVES | Black rot | Provisional; disease identity requires confirmation |
| HEALTHY LEAVES | Healthy | Folder-derived label |

There is no cedar-rust ground truth. Predictions of cedar rust remain errors;
its zero-support precision/recall/F1 entries in machine-readable tables must not
be read as a measurement of cedar-rust detection. Headline macro-F1 averages
over the three represented classes. The four-class average is saved separately.
Results describe the available subset, not the complete intended dataset or
field deployment performance. Missing files and ambiguous labels need resolution
before this can be considered a definitive benchmark.

## Results

| Model | Correct / total | Accuracy | Macro-F1 (3 present classes) | Macro-F1 (all 4 classes) |
| --- | --- | --- | --- | --- |
| Baseline | 63 / 260 | 24.23% | 0.2851 | 0.2138 |
| Background augmented | 63 / 260 | 24.23% | 0.2495 | 0.1871 |

Always predicting scab would achieve 45.00% accuracy on this subset. Neither
classifier exceeds that majority-class reference.

| Model | Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- | --- |
| Baseline | Scab | 0.5676 | 0.1795 | 0.2727 | 117 |
| Baseline | Rot → black rot (provisional) | 0.3953 | 0.1700 | 0.2378 | 100 |
| Baseline | Healthy | 0.2451 | 0.5814 | 0.3448 | 43 |
| Background augmented | Scab | 0.5814 | 0.2137 | 0.3125 | 117 |
| Background augmented | Rot → black rot (provisional) | 0.3182 | 0.0700 | 0.1148 | 100 |
| Background augmented | Healthy | 0.2067 | 0.7209 | 0.3212 | 43 |

Confusion matrices use rows = true folder-derived class and columns = predicted
class, ordered scab, black rot, cedar rust, healthy:

| Baseline true class | Scab | Black rot | Cedar rust | Healthy |
| --- | --- | --- | --- | --- |
| Scab | 21 | 23 | 30 | 43 |
| Rot | 16 | 17 | 33 | 34 |
| Cedar rust (absent) | 0 | 0 | 0 | 0 |
| Healthy | 0 | 3 | 15 | 25 |

| Augmented true class | Scab | Black rot | Cedar rust | Healthy |
| --- | --- | --- | --- | --- |
| Scab | 25 | 6 | 27 | 59 |
| Rot | 15 | 7 | 18 | 60 |
| Cedar rust (absent) | 0 | 0 | 0 | 0 |
| Healthy | 3 | 9 | 0 | 31 |

## Confidence and failures

| Model | Mean confidence | Mean on correct | Mean on errors | Predictions ≥ 0.90 | Errors among those |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0.7662 | 0.7458 | 0.7728 | 78 | 64 (82.05%) |
| Background augmented | 0.6265 | 0.5868 | 0.6392 | 28 | 24 (85.71%) |

Softmax confidence is uncalibrated. Errors have higher mean confidence than
correct predictions for both models; a 0.90 confidence cutoff does not establish
reliability. The augmented model predicts healthy for 119 of 217 diseased-folder
images. Its highest-confidence errors predominantly show rot-folder leaves
predicted healthy on pink backgrounds. This visual observation does not establish
the cause of failure or independently confirm a disease diagnosis.

Excluding the provisional rot category entirely, accuracy over the remaining
160 scab/healthy images is 28.75% for baseline and 35.00% for augmented. Thus,
the weak result persists without relying on the rot mapping.

## Reproduction and outputs

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m appleleaf.cli.evaluate_kashmir `
  --checkpoint artifacts\apple_baseline_checkpoint.pt `
  --output-dir artifacts\kashmir_evaluation\baseline `
  --exclude-label-conflicts --exclude-missing-lfs

.venv\Scripts\python.exe -m appleleaf.cli.evaluate_kashmir `
  --checkpoint artifacts\background_augmented\apple_background_augmented_checkpoint.pt `
  --output-dir artifacts\kashmir_evaluation\background_augmented `
  --exclude-label-conflicts --exclude-missing-lfs
```

Use an empty output directory for each run. By default, conflicting labels and
missing image contents stop evaluation; the explicit exclusion flags save audit
tables. Other corrupt images always stop evaluation. If rot is independently
confirmed as black rot, use `--rot-label-status confirmed` to record that status.
The installed console entry point is `appleleaf-evaluate-kashmir`.

Each model directory contains metrics JSON (including confidence distributions,
bins, class mapping, and checkpoint SHA-256), per-class CSV, confusion-matrix CSV
and PNG, all per-image probabilities, compressed predictions, manifest with image
SHA-256, both exclusion tables, all failures, and the 12 highest-confidence failure
examples as CSV and PNG. CSV/JSON/text snapshots are preserved under
`results/2026-10-08_kashmir/`; generated images remain under `artifacts/`.

Three focused unit tests cover four-way scoring with an absent class, explicit
conflict exclusion, and missing LFS detection. Both checkpoint evaluations ran
successfully, and the augmented failure sheet was visually inspected.

## Decision and next steps

On 2026-10-08, both checkpoints exactly reproduced their original PlantVillage
test results: 90.32% baseline and 83.79% background augmented, with all saved
predictions and probabilities identical. Checkpoint hashes also match this
evaluation. See [PLANTVILLAGE_REPRODUCTION.md](PLANTVILLAGE_REPRODUCTION.md).
Missing-image recovery is now complete. Independent verification of Kashmir's
disease labels remains pending; demo integration remains paused. The subsequent
[label audit](KASHMIR_LABEL_AUDIT.md) documents additional conflicts.

Neither classifier is validated for reliable external disease identification.
Retain both frozen checkpoints as references. Missing LFS recovery is complete;
resolve the disease labels and conflicting assignments before reporting a
complete Kashmir benchmark; do not tune
these models on this evaluation set.

Next, evaluate the chosen lesion checkpoints on an external dataset with actual
lesion masks and documented independence from training. The Kashmir folders have
classification labels only and cannot supply segmentation Dice/IoU. Then combine
classification and segmentation in the demo, exposing their separate outputs and
the measured limits of disease classification. External segmentation and demo
integration have not been performed in this step.

yet to be done 
