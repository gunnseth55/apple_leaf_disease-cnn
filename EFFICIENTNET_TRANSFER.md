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
results; EfficientNet external generalization has not yet been evaluated.

Accuracy and macro-F1 were independently recalculated from the saved
`test_predictions.npz` and matched `metrics.json` exactly. All 475 labels and
predictions also match `test_predictions.csv`. This verification used saved
outputs only and did not rerun training or model inference.

Tracked configuration, metrics and history snapshots are in
`results/2026-10-10_efficientnet_b0/`. Weights, per-image predictions, manifests
and figures remain under `artifacts/efficientnet_b0/`.

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
