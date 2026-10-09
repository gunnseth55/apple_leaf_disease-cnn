# EfficientNet-B0 classifier transfer learning

Code prepared on 2026-10-10. The user completed the 15-epoch training run on CPU;
Codex did not run any training epochs. Saved results were verified on 2026-10-10.

## Completed PlantVillage results

The selected checkpoint is **epoch 11**, with validation macro-F1 **1.0000**.
Test accuracy is **99.79% (474/475)** and macro-F1 is **0.9981701493**.

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Apple scab | 1.0000 | 0.9895 | 0.9947 | 95 |
| Black rot | 1.0000 | 1.0000 | 1.0000 | 93 |
| Cedar apple rust | 1.0000 | 1.0000 | 1.0000 | 41 |
| Healthy | 0.9960 | 1.0000 | 0.9980 | 246 |

The only error was scab predicted as healthy:
`fb942296-ba33-40e5-ada0-6700365cf71d___FREC_Scab 2919.JPG`.
Test accuracy increased from the original CNN's 90.32% to 99.79%, a gain of
9.47 percentage points on the same split. These are in-distribution PlantVillage
results; the subsequent Kashmir evaluation below shows weak external performance.

Accuracy and macro-F1 were independently recalculated from the saved
`test_predictions.npz` and matched `metrics.json` exactly. All 475 labels and
predictions also match `test_predictions.csv`. This verification used saved
outputs only and did not rerun training or model inference.

Tracked configuration, metrics and history snapshots are in
`results/2026-10-10_efficientnet_b0/`. Weights, per-image predictions, manifests
and figures remain under `artifacts/efficientnet_b0/`.

## Saved best checkpoint and settings

Verified against `training_history.csv`: epoch 11 is the earliest epoch with the
highest validation macro-F1 (1.0000). The completed training command already saved
and restored this checkpoint; it is not the last epoch's weights. The checkpoint
was retained without rewriting its weights or rerunning training.

| Setting | Recorded value |
| --- | --- |
| Checkpoint name | `apple_efficientnet_b0_checkpoint.pt` |
| Checkpoint path | `artifacts/efficientnet_b0/apple_efficientnet_b0_checkpoint.pt` |
| Architecture / initial weights | EfficientNet-B0 / ImageNet-1K V1 |
| Selection | Highest validation macro-F1; earliest epoch wins ties |
| Selected epoch / validation macro-F1 | 11 / 1.0000 |
| Preprocessing | PIL RGB; full-image bicubic resize; ToTensor; normalization; no crop |
| Image size | 224 x 224 |
| Normalization mean (RGB) | `[0.485, 0.456, 0.406]` |
| Normalization std (RGB) | `[0.229, 0.224, 0.225]` |
| Optimizer / weight decay | AdamW / 0.0001 |
| Warmup LR | Head 0.001; backbone frozen |
| Fine-tuning initial LR | Backbone 0.0001; head 0.0003 |
| LR schedule | CosineAnnealingLR over 12 fine-tuning epochs |
| LR at selected epoch 11 | Backbone 0.00003705904774487398; head 0.0001111771432346219 |
| Completed epochs | 15 total: 3 head warmup + 12 full fine-tuning |
| Batch size / seed | 16 / 42 |
| Class mapping | 0 = Apple scab; 1 = Black rot; 2 = Cedar apple rust; 3 = Healthy |

Machine-readable metadata, including the checkpoint SHA-256, is saved in
`artifacts/efficientnet_b0/checkpoint_metadata.json` and the tracked snapshot
`results/2026-10-10_efficientnet_b0/checkpoint_metadata.json`. Existing checkpoint
metadata includes the selection, optimizer, learning rates, epochs and class
mapping; the companion record also makes normalization values explicit.

## User-run evaluation results, 2026-10-10

The separate PlantVillage evaluation reproduced 474/475 correct (99.79%) and
macro-F1 0.9981701493. Saved ordered labels and predictions are identical to
training's final test output; maximum probability difference is 2.2352e-08.
Mean confidence is 0.99736; the single error has confidence 0.71109. All 472
predictions with confidence >=0.90 are correct on this split.

The frozen model achieved **68/260 correct (26.15%)** on the original Kashmir
available subset, below its 45.00% majority-class reference. Present-class
macro-F1 is **0.2248473493**; the all-four-class macro-F1 is **0.1686355120**.
**The total Kashmir score is preliminary pending independent verification of
rot identity and the source labels.** The class breakdown and representative
visual review are in [KASHMIR_EFFICIENTNET_REVIEW.md](KASHMIR_EFFICIENTNET_REVIEW.md).
The terminal classification report includes the absent cedar-rust class in its
macro average, explaining the two macro-F1 values. Cedar-rust detection is not
measured by these zero-support entries.

| Kashmir folder-derived class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Scab | 0.3765 | 0.2735 | 0.3168 | 117 |
| Rot mapped to black rot (provisional) | 0.5000 | 0.0200 | 0.0385 | 100 |
| Healthy | 0.2000 | 0.7907 | 0.3192 | 43 |

Mean Kashmir confidence is 0.84403, with errors more confident on average
(0.85343) than correct predictions (0.81749). Of 141 predictions with confidence
>=0.90, **107 are errors (75.89%)**. The model predicts healthy for 136/217
diseased-folder images. Excluding provisional rot, scab/healthy accuracy is
66/160 (41.25%); weak performance persists without that mapping.

Saved manifests confirm the same 260-image inventory as the original CNN
evaluation; both EfficientNet evaluations record the preserved checkpoint hash.
This run excludes 47 missing LFS images and six exact label-conflict rows. It
does not use the later recovered 283-image candidate inventory or remove all
subsequently identified visual conflicts. Kashmir labels remain unverified.

Decision: retain EfficientNet as the stronger PlantVillage reference, but do not
claim reliable external diagnosis or tune it on Kashmir. The gap is consistent
with distribution mismatch and label problems; these scores do not establish
their relative causes. Original evaluation outputs are in
`artifacts/efficientnet_b0_evaluation_full/` and
`artifacts/kashmir_evaluation/efficientnet_b0/`. CSV/JSON/text snapshots are in
`results/2026-10-10_efficientnet_b0/{plantvillage_evaluation,kashmir}/`.
The user ran both evaluations; Codex verified saved outputs only.

## Data and experiment

Only the four Apple classes from PlantVillage colour images enter training.
The existing decoded-RGB exact-image deduplication and stratified 70/15/15 split
with seed 42 are unchanged: 2,214 train, 475 validation, 475 test in the cached
version 3 dataset. Split manifests include image hashes and are saved per run.
No Kashmir images, lesion-segmentation images, synthetic images, or healthy
external benchmark images are used to train or select this classifier.

The starting model is torchvision EfficientNet-B0 with ImageNet-1K V1 weights
and a replacement four-class linear head. First use downloads the pretrained
weights through torchvision to the usual PyTorch cache.

- Input: RGB, full-image bicubic resize to 224 x 224, ToTensor, ImageNet mean
  `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
- Full-image resize deliberately replaces the official weights' resize/center
  crop sequence, preserving peripheral leaf lesions and overlay alignment.
- Training-only augmentation: horizontal/vertical flips, rotation up to 15
  degrees, mild brightness/contrast/saturation changes. Backgrounds remain original.
- Loss: cross entropy with inverse-frequency weights from the training split.
- Epochs 1–3: frozen backbone, including batch-normalization statistics and
  stochastic depth; train the head using AdamW, learning rate 0.001.
- Epochs 4–15: restore the best warmup checkpoint, then fine-tune all layers;
  backbone LR 0.0001, head LR 0.0003, cosine decay, weight decay 0.0001.
- Select the highest validation macro-F1 across both phases, with earliest
  epoch winning ties. Test inference occurs only after training finishes.

This is a transfer-learning experiment, not an architecture-only comparison:
pretraining, resolution, augmentation, optimizer, and training schedule differ
from the original CNN. The identical split supports a useful reference comparison,
but does not isolate the effect of architecture. Exact deduplication does not
guarantee independence of related views or near duplicates.

## Run it yourself

From the project root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m appleleaf.cli.train_efficientnet `
  --dataset-path C:/Users/gunn/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3 `
  --artifacts-dir artifacts/efficientnet_b0 `
  --epochs 15 --warmup-epochs 3 --batch-size 16 --device cpu
```

The installed environment currently has CPU-only PyTorch. The command will take
longer than GPU training. Use `--device cuda` only in an environment with a
working CUDA-enabled PyTorch installation. Lower `--batch-size` if memory is tight.

For a split/configuration check without model downloads or any epochs, append
`--dry-run`. This reads the images to verify readability and exact deduplication.
It creates no training outputs. The training command rejects a nonempty output
folder; choose a new `--artifacts-dir` for another run.

After reinstalling this project, the equivalent console command is
`appleleaf-train-efficientnet`. Module invocation above works without reinstalling.

## Outputs and evaluation

The completed run's `artifacts/efficientnet_b0/` contains:

- `apple_efficientnet_b0_checkpoint.pt` (best weights, architecture, preprocessing,
  class mapping, configuration, validation score; test scores after completion).
- `run_config.json`, training/validation/test manifests, `training_history.csv`.
- `metrics.json`, `test_predictions.csv`, `test_predictions.npz`.
- Training curves and test confusion matrix.

The best checkpoint and history are saved after each completed epoch. An
interrupted run can leave a usable best checkpoint but no final test results;
the command does not implement automatic training resume. Look for
`training_complete=True` in checkpoint metadata to distinguish a completed run.

Evaluate the saved checkpoint with the existing evaluation CLI:

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m appleleaf.cli.evaluate `
  --checkpoint artifacts/efficientnet_b0/apple_efficientnet_b0_checkpoint.pt `
  --dataset-path C:/Users/gunn/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3 `
  --output-dir artifacts/efficientnet_b0_evaluation --device cpu
```

Evaluation, background audits, Grad-CAM explanations, and Kashmir evaluation
now read the checkpoint architecture and preprocessing. Legacy CNN checkpoints
retain their original model and unnormalized 128 x 128 preprocessing. Kashmir
remains evaluation-only, with its unresolved label limitations; a stronger
PlantVillage score alone will not establish reliable field disease diagnosis.

The PlantVillage evaluation command now saves `metrics.json` (accuracy,
macro-F1, per-class precision/recall/F1 and confidence summary),
`per_class_metrics.csv` (including class-wise recall), `classification_report.txt`,
`confusion_matrix.csv`, `confusion_matrix.png`, `confidence_bins.csv`,
`confidence_distribution.png`, `predictions.csv` and `test_predictions.npz`.
Confidence is maximum softmax probability and is uncalibrated. The confidence
histogram distinguishes correct predictions from errors. Choose a new/empty
output folder for each evaluation. Codex prepared these outputs but did not
run model evaluation; the user runs the command above.
