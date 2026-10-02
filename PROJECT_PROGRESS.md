# Project Progress Log

This document records how the Apple leaf disease classifier developed, what was
learned from each experiment, and why each change was made. Dates are
reconstructed from the project files and experiment sessions.

## 2026-09-22 — Initial notebook baseline

### Goal

Build an initial four-class Apple leaf disease classifier using the PlantVillage
dataset.

### Work completed

- Explored the dataset in `apple_leaf_baseline.ipynb`.
- Selected four Apple classes: Apple scab, black rot, cedar apple rust, and
  healthy.
- Built a compact CNN with four convolutional blocks, global average pooling,
  dropout, and a four-class linear output layer.
- Trained directly on complete RGB images.

### Initial limitation

The network received only image-level class labels. It was never told which
pixels represented the leaf, disease symptoms, or background. It could therefore
learn both legitimate leaf features and unintended correlations in the scene.

## 2026-10-01 — Reproducible baseline pipeline

### Goal

Move the experiment out of notebook state and make training and evaluation
repeatable.

### Changes

- Created the installable `appleleaf` package under `src/appleleaf/`.
- Added deterministic data preparation and stratified train, validation, and
  test splits.
- Added exact-image duplicate removal before splitting.
- Added weighted cross-entropy to account for class imbalance.
- Added checkpointing based on validation macro-F1.
- Added command-line tools for dataset download, training, and evaluation.
- Saved histories, predictions, confusion matrices, and checkpoints under
  `artifacts/`.

### Baseline result

- Best epoch: 14
- Test accuracy: 90.32%
- Test macro-F1: 0.9113

The baseline performed well on the standard held-out images, but this alone did
not establish that it was using disease symptoms rather than dataset-specific
background cues.

## 2026-10-01 — Controlled-background audit

### Question

Does the baseline rely on the leaf itself, or does it exploit background
information correlated with the labels?

### Changes

- Matched every colour image with its PlantVillage segmented version.
- Derived a foreground mask from the segmented image.
- Added controlled validation conditions that retain the original leaf while
  replacing the background with white, blue, or black.
- Added prediction-change rate and mean-confidence measurements.

### Results

| Condition | Accuracy | Macro-F1 | Prediction change |
| --- | ---: | ---: | ---: |
| Original | 89.05% | 0.8918 | 0.00% |
| White | 46.95% | 0.2948 | 51.16% |
| Blue | 19.79% | 0.0826 | 76.00% |
| Black | 19.79% | 0.0873 | 76.84% |

### Analysis

The large accuracy collapse showed that predictions were highly sensitive to
background changes. Blue and black were severe distribution shifts, so these
results alone could not prove background shortcut learning. Mask boundaries and
unfamiliar colours could also cause the failure.

### Decision

Keep the existing CNN as the baseline. A larger model might learn the same
shortcut more effectively, so changing architecture was postponed until the
data issue could be tested directly.

## 2026-10-02 — Background-only diagnostic

### Question

Can the baseline predict the class after leaf colour, texture, and disease
symptoms have been removed?

### Changes

- Added a `background_only` audit condition.
- Preserved the original scene outside the segmentation mask.
- Replaced the leaf region with mid-grey.
- Added the majority-class accuracy to the audit output.

### Result

- Background-only accuracy: 63.16%
- Background-only macro-F1: 0.5160
- Majority-class baseline: 51.79%

On 475 validation images, the model classified 300 correctly using the retained
background, silhouette, and mask-boundary information. Always predicting the
majority class would classify 246 correctly.

### Interpretation

This provided direct evidence that the trained baseline used information beyond
disease appearance. Because the masked image retains the leaf outline and an
artificial boundary, the result should be described as background-related
leakage rather than proof that background colour alone is responsible.

## 2026-10-02 — Randomized-background training

### Goal

Test whether the same architecture becomes more robust when the background is
made unreliable during training.

### Experimental control

The CNN architecture, class weights, optimizer, seed, split, image size, and
epoch count remained unchanged. Only the training-image backgrounds changed.
Validation and test images remained original.

### Implementation

- Added `RandomBackgroundDataset` in `src/appleleaf/datasets.py`.
- Added the `--randomize-backgrounds` option to the existing training command.
- Retained the original background for 25% of training samples.
- For the remaining samples, replaced the background with one of:
  - black, white, or blue;
  - an arbitrary RGB colour;
  - a smooth randomly generated texture.
- Sampled a new background whenever an image was loaded, allowing backgrounds
  to vary between epochs.
- Stored the new run separately in `artifacts/background_augmented/`.

### First augmentation trial

The first version sampled arbitrary colours and textures but did not explicitly
include exact black. It improved white and blue robustness, but black-background
accuracy remained only 25.68%. Uniform RGB sampling almost never generates an
exact or near-black image.

### Revision

Added black, white, and blue as explicit class-independent training backgrounds,
alongside arbitrary colours and textures, and retrained from the same seed.

### Final augmented-model results

- Best epoch: 12
- Test accuracy: 83.79%
- Test macro-F1: 0.7953

| Audit condition | Baseline | Background augmented | Change |
| --- | ---: | ---: | ---: |
| Original | 89.05% | 84.63% | -4.42 points |
| White | 46.95% | 80.84% | +33.89 points |
| Blue | 19.79% | 81.26% | +61.47 points |
| Black | 19.79% | 80.42% | +60.63 points |
| Background only | 63.16% | 51.79% | -11.37 points |

### Conclusion

The same small CNN became dramatically more stable under controlled background
changes, while its background-only performance fell to the majority baseline.
This supports the hypothesis that the main robustness problem was background
shortcut learning encouraged by the training data, rather than architecture
alone.

The augmented model currently trades approximately 4.4 validation-accuracy
points on original images for much stronger background invariance. Architecture
changes should therefore be evaluated only after tuning this trade-off and
testing on realistic external photographs.

## Next planned work

- [ ] Evaluate both checkpoints on external or field images with natural
   backgrounds.
- [ ] Visually inspect more masks for missing leaf regions or strong boundary
   artifacts.
- [x] Add Grad-CAM comparisons for the baseline and augmented models (completed
  2026-10-03).
- [ ] Tune the probability and realism of background augmentation to recover
   original-image accuracy.
- [ ] Only then compare this CNN with a pretrained architecture such as ResNet18 or
   EfficientNet-B0.

## 2026-10-03 — Per-image prediction explanations

### Goal

Make every selected test prediction inspectable: show which image was evaluated,
where the CNN found influential evidence, and how its class scores led to the
final prediction.

### Changes

- Added `appleleaf-explain`, which operates on explicit indices from the
  deterministic held-out test split.
- Added Grad-CAM using the final convolutional feature tensor.
- Added a three-panel explanation figure containing:
  - the exact test image with the segmentation boundary;
  - a Grad-CAM overlay showing influential regions;
  - probabilities for all four classes.
- Added prediction correctness, confidence, source path, and inside/outside-leaf
  attention fractions to `explanations.csv`.
- Generated matched examples for test indices 0 through 7 using both models.

### Revised balanced observation

The first examples were healthy-class-heavy, so the selection was replaced with
two deterministic examples from each of the four classes. Grad-CAM was also
changed to use the feature map before the final max-pooling layer, increasing
the native explanation resolution from 8x8 to 16x16.

Across the eight balanced examples, the baseline placed an average of 51.05% of
its positive Grad-CAM intensity inside the leaf. The background-augmented model
placed 75.06% inside. This small example is descriptive rather than a complete
statistical evaluation, but it agrees with the controlled-background audit.

One particularly informative healthy image at test index 1 showed only 6.84%
inside-leaf attention for the baseline; its strongest influence appeared in the
background and shadow. The augmented model moved 72.27% of its attention inside
the leaf on the same image, although it incorrectly predicted black rot. This
also demonstrates why an explanation must be shown together with correctness
and class probabilities rather than treated as proof that the model is right.

### Interpretation limitation

Grad-CAM is derived from gradients of a class score with respect to the final
convolutional features. Because this model's final feature map is spatially
coarse, the heatmap identifies influential regions, not exact disease pixels or
clinically verified lesion borders.

### Background-transformation preview

Added `appleleaf-preview-augmentation` to display class-balanced original
training images beside their leaf masks and white, blue, black, and random
texture backgrounds. This separates two different visual questions:

- the preview shows how training inputs are altered;
- Grad-CAM shows where a trained model finds influential evidence on an
  untouched test image.

### Requirement for exact lesion detection

Inspection of the local PlantVillage data confirmed that its `segmented`
variant labels the whole leaf against the background. It does not provide
ground-truth masks around disease lesions. Exact lesion localization cannot be
trained or objectively scored from the current class labels. The next disease
localization phase must introduce pixel-level lesion annotations and evaluate a
segmentation model with metrics such as intersection-over-union and Dice score.

## 2026-10-03 — Separate lesion-segmentation pipeline

### Goal

Move from image-level classification explanations to trainable and measurable
pixel-level disease-lesion detection without mixing the new task into the
existing classifier.

### Changes

- Created the independent `src/appleleaf_lesion/` package.
- Implemented a binary U-Net with an ImageNet-pretrained ResNet34 encoder.
- Implemented equally weighted Dice and focal loss.
- Added paired image/mask loading with synchronized flips and ImageNet
  normalization.
- Added best-checkpoint selection using validation Dice.
- Added Dice, IoU, pixel precision, and pixel recall metrics.
- Added a separate evaluation command producing ground-truth masks, probability
  maps, and red lesion overlays.
- Documented the required train/validation/test manifest format in
  `LESION_DETECTION.md`.

### Verification

The complete train, checkpoint-load, evaluation, and visualization workflow was
smoke-tested on generated geometric masks. This validates the software path only;
the resulting smoke-test metrics have no biological meaning.

### Current blocker

The cached PlantVillage data supplies whole-leaf segmentation masks but no
pixel-level lesion masks. Real lesion training must not start until a suitable
annotated dataset is obtained and its mask semantics are visually verified.

## 2026-10-03 — Lesion dataset acquisition and validation

### Dataset selected

Downloaded the PlantVillage Apple Synthetic Segmentation Dataset from Zenodo
(DOI `10.5281/zenodo.18659728`). Both published MD5 checksums were verified
before extraction.

### Split design

- Training: 30 manually annotated real pairs plus 300 synthetic pairs
- Validation: 15 manually annotated real pairs
- Test: 30 manually annotated real pairs
- Each split is balanced across scab, black rot, and cedar rust

Synthetic samples are restricted to training so final performance is measured
on real PlantVillage leaves with manually created lesion annotations.

### Mask-semantic correction

Visual and numeric inspection found three mask regions rather than a conventional
binary mask:

- black: image background;
- white: unaffected leaf;
- saturated red, blue, or green: disease lesion.

The initial generic nonzero-mask interpretation would have mislabeled the entire
leaf as diseased. The loader was corrected to isolate only saturated lesion
colours while retaining support for conventional black/white lesion masks.

### Verification

Generated reproducible manifests with 330 training, 15 validation, and 30 test
pairs. A one-epoch pretrained ResNet34 U-Net smoke test completed end to end.
Its test Dice of 0.1563 is intentionally not treated as a result: the overlay
correctly finds lesions but contains extensive false positives after only one
epoch. The test confirms image/mask alignment, loss calculation, checkpointing,
metrics, and visualization before full training.

## Maintenance convention

For every future experiment, append a dated section containing:

- the question or hypothesis;
- the exact code or data change;
- controlled variables;
- metrics and artifact locations;
- interpretation and limitations;
- the decision made from the result.
