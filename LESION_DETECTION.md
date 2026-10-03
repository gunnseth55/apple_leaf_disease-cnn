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
- Best checkpoint selected by validation Dice score
- Probability threshold calibrated on the validation split after training
- Dice, intersection-over-union, pixel precision, and pixel recall evaluation

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
