# Lesion Detection Pipeline

This pipeline is intentionally separate from the existing image-classification
and background-bias experiments. Its task is binary pixel-level segmentation:
predict which pixels belong to a disease lesion.

## Selected dataset

The project uses the **PlantVillage Apple Synthetic Segmentation Dataset**, DOI
`10.5281/zenodo.18659728`. It contains manually annotated real images for Apple
scab, black rot, and cedar rust, plus aligned synthetic image/mask pairs.

- Real train: 30 images (10 per disease)
- Real validation: 15 images (5 per disease)
- Real test: 30 images (10 per disease)
- Synthetic training addition: 300 images (100 per disease)

Synthetic images are used only for training. Validation and test metrics use
manually annotated real images.

## Dataset contract

The current PlantVillage `segmented` images are whole-leaf masks and must **not**
be used as lesion ground truth. Supply three CSV files in one manifest folder:

`lesion_manifests` is not generated automatically and is not included in the
repository. It can be created only after obtaining images with genuine
pixel-level disease-lesion annotations.

```text
lesion_manifests/
├── train.csv
├── val.csv
└── test.csv
```

Each CSV must contain absolute or working-directory-relative paths:

```csv
image_path,mask_path
C:\data\images\image_001.jpg,C:\data\masks\image_001.png
```

The loader accepts conventional binary masks (black background and white lesion)
and the Zenodo Apple masks, where black is background, white is unaffected leaf,
and saturated red, green, or blue marks the lesion. Images and masks belonging
to the same source must never be split across train, validation, and test
manifests.

For the Zenodo masks specifically, black is image background, white is
unaffected leaf, and lesions are red for black rot, blue for cedar rust, or green
for scab. The loader converts only the saturated coloured pixels into the binary
lesion target.

## Architecture and objective

- U-Net decoder
- ImageNet-pretrained ResNet34 encoder
- 256x256 input by default
- Dice loss + focal loss with equal weights
- AdamW optimizer
- Strong training-only augmentation: paired flips, rotation, translation,
  scaling, and shear, with image-only colour jitter
- Best checkpoint selected by validation Dice score
- Probability threshold calibrated on the validation split after training
- Dice, intersection-over-union, pixel precision, and pixel recall evaluation
- Per-image and per-disease evaluation, including mean/median image Dice and
  identification of the five worst-performing test images

## Train

For the Zenodo dataset, generate manifests while retaining its official split:

```powershell
appleleaf-lesion-prepare-zenodo `
  --dataset-root data\lesion_zenodo\restricted\Restricted_Plant_Village_Dataset `
  --synthetic-root data\lesion_zenodo\synthetic\Synthetic_Dataset `
  --output-dir lesion_manifests
```

Synthetic pairs are appended only to `train.csv`. Validation and testing retain
the dataset's manually annotated real images, preventing optimistic evaluation
on generated samples.

```powershell
appleleaf-lesion-train `
  --manifest-dir lesion_manifests `
  --epochs 30 `
  --batch-size 8
```

To run a controlled source-balancing experiment without the hard-negative rows,
override only the training manifest and sample 30% real versus 70% synthetic
images per epoch:

```powershell
appleleaf-lesion-train `
  --manifest-dir lesion_manifests `
  --train-manifest lesion_manifests\train_before_hard_negatives.csv `
  --real-sampling-fraction 0.30 `
  --epochs 30 `
  --batch-size 8 `
  --artifacts-dir artifacts\lesion_detection_balanced_30_70
```

Sampling uses replacement and retains 330 draws per epoch. The expected source
mix is 99 real and 231 synthetic draws, while validation and test data remain
unchanged.

To test a small, controlled hard-negative exposure without changing the number
of optimization steps, train from `train.csv` and reserve 5% of the same 330
draws for healthy hard negatives:

```powershell
appleleaf-lesion-train `
  --manifest-dir lesion_manifests `
  --train-manifest lesion_manifests\train.csv `
  --real-sampling-fraction 0.30 `
  --hard-negative-sampling-fraction 0.05 `
  --samples-per-epoch 330 `
  --epochs 30 `
  --batch-size 8 `
  --artifacts-dir artifacts\lesion_detection_real30_synthetic70_hardneg05
```

Here, the 30/70 ratio applies within the 95% lesion-image portion of each
epoch. The expected overall probability mass is therefore 28.5% real lesion,
66.5% synthetic lesion, and 5% healthy hard negative. Validation and test
remain the same manually annotated real lesion images used by the preceding
30/70 experiment.

To preserve all lesion draws while adding a separate healthy empty-mask
objective, use the original lesion-only manifest and select the original
hard-negative rows through an auxiliary loader:

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
```

The auxiliary loader includes only `source=hard_negative` rows, applies training
augmentation, and cycles across lesion batches. Each optimizer step combines
the usual lesion Dice/focal loss and `0.05 * healthy BCE`; the healthy targets
must be empty. The loss weight is not a sampling fraction. The 50 held-out
healthy benchmark images must never be included in this training loader.

The completed run restored epoch 18 and calibrated threshold 0.55. Test Dice
was 0.6933; healthy FP pixel rate was 0.7709%. It did not replace pure 30/70
or hardneg05. Full results and reproduction commands are in
[the tracked experiment snapshot](results/2026-10-07_dualhealthy05/README.md).

The pretrained encoder weights are downloaded by PyTorch on first use. Use
`--no-pretrained` only for offline smoke tests, not for the main experiment.

## Evaluate and inspect lesion locations

```powershell
appleleaf-lesion-evaluate `
  --checkpoint artifacts\lesion_detection\unet_resnet34_lesion_checkpoint.pt `
  --manifest lesion_manifests\test.csv `
  --output-dir artifacts\lesion_evaluation
```

The evaluation command creates `metrics.json` and side-by-side figures showing
the image, ground-truth lesion mask, predicted probability map, and red lesion
overlay. The overlay and metrics both use the validation-calibrated threshold
stored in the checkpoint. For older checkpoints, it can be overridden with
`--threshold`. Unlike Grad-CAM, this output can be evaluated as pixel-level
lesion localization because it is compared with an annotated ground-truth mask.
It also writes `per_image_metrics.csv` and `per_disease_metrics.csv`. The latter
reports pooled metrics within each disease as well as mean, median, and standard
deviation of per-image Dice scores.
