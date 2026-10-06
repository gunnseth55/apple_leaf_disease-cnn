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

## 2026-10-04 — Paired augmentation and hard-negative lesion experiment

### Question

Test whether stronger paired augmentation improves lesion segmentation, then
whether healthy leaves with confusing structures reduce false positives on
stem ends, leaf tips, folds, and holes.

### Changes

- Extended training-only augmentation with paired flips, rotation, translation,
  scale, and shear. The same geometry is applied to each image and mask, using
  nearest-neighbour interpolation for masks.
- Added image-only brightness, contrast, saturation, and hue jitter.
- Selected 40 confidently healthy Plant Pathology 2020 images containing stems,
  folds, holes, shadows, sunlight, and natural backgrounds.
- Created aligned all-black lesion masks for these healthy hard negatives.
- Expanded `lesion_manifests/train.csv` from 330 to 370 pairs. Validation and
  test manifests were unchanged.
- Added reusable scripts under `scripts/` for candidate collection, empty-mask
  creation, and duplicate-safe manifest updates.

### Results

The original 30-epoch model scored Dice 0.6806, IoU 0.5159, precision 0.6059,
and recall 0.7763 on the fixed test split.

With paired augmentation, the best checkpoint was restored from epoch 20. Its
validation Dice was 0.6603, increasing to 0.6653 after calibrating the threshold
to 0.70. Test Dice improved to 0.6873, IoU to 0.5236, and precision to 0.6281;
recall decreased to 0.7589. Results are stored in
`artifacts/lesion_evaluation_augmented`.

With the 40 hard negatives added, the best checkpoint was restored from epoch
27. Validation Dice was 0.6617 and remained 0.6618 after threshold calibration
to 0.55. Test Dice fell to 0.6485, IoU to 0.4798, and precision to 0.5584, while
recall increased to 0.7733. The checkpoint and evaluation are stored in
`artifacts/lesion_detection_hard_negatives` and
`artifacts/lesion_evaluation_hard_negatives`.

### Visual inspection and decision

Despite the aggregate regression, the hard-negative model stopped marking the
previously observed stem ends and leaf tips as lesions. It also stopped marking
holes in several examples, although one hole remained a false positive. This
shows that targeted healthy examples addressed the intended error type, but the
current mixture or training setup reduced overall segmentation quality.

The augmented model remains the preferred general checkpoint because it has the
best test Dice, IoU, and precision. The hard-negative checkpoint is retained as
experimental evidence rather than replacing it. A future experiment should
seek the same structural false-positive improvement with less effect on lesion
segmentation, for example through fewer or more tightly matched hard negatives
or controlled sampling.

### Per-image and per-disease diagnosis

Extended lesion evaluation to preserve the existing pooled metrics while also
writing `per_image_metrics.csv` and `per_disease_metrics.csv`. Reporting now
includes mean, median, and standard deviation of per-image Dice plus the five
worst-performing images. Unit tests cover the per-sample calculation.

On the preferred augmented checkpoint, pooled test Dice remained 0.6873. Mean
per-image Dice was 0.6960, median 0.6773, and standard deviation 0.1137. The
per-disease results were:

- black rot: pooled Dice 0.8314; mean-image Dice 0.7988;
- cedar rust: pooled Dice 0.6384; mean-image Dice 0.6346;
- scab: pooled Dice 0.6574; mean-image Dice 0.6545.

Cedar rust supplied four of the five worst individual images, with Dice scores
from 0.4728 to 0.5844; the remaining image was scab at 0.5113. This establishes
cedar rust as the clearest weakness hidden by the previous single global score.
Outputs are stored in `artifacts/lesion_evaluation_augmented_per_disease`.

## 2026-10-04 — 30/70 real-synthetic lesion sampling

### Question

Does increasing the probability of manually annotated real training images from
their natural 9.1% share to 30% improve real-image lesion segmentation,
especially for cedar rust and scab, without sacrificing black-rot performance?

### Controlled change

- Trained from `lesion_manifests/train_before_hard_negatives.csv`, containing 30
  real and 300 synthetic lesion pairs balanced across the three diseases.
- Used weighted sampling with replacement so each 330-sample epoch drew 30%
  real and 70% synthetic examples in expectation.
- Kept the existing paired geometric and colour augmentation, ResNet34 U-Net,
  Dice/focal loss, seed 42, 256-pixel input size, batch size 8, 30 epochs, and
  fixed validation and test splits.
- Excluded the 40 healthy hard negatives so source sampling was the only changed
  experimental variable.

### Results

The best checkpoint was restored from epoch 25. Validation Dice increased from
0.6809 at the default threshold to 0.6903 after calibrating the threshold to
0.70. On the fixed 30-image real test split, pooled Dice improved from 0.6873
to 0.7043, IoU from 0.5236 to 0.5435, and precision from 0.6281 to 0.6475;
recall increased from 0.7589 to 0.7719.

| Disease | Previous Dice | 30/70 Dice | Change | 30/70 precision | 30/70 recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Black rot | 0.8314 | 0.8317 | +0.0003 | 0.8105 | 0.8540 |
| Cedar rust | 0.6384 | 0.6657 | +0.0273 | 0.5771 | 0.7863 |
| Scab | 0.6574 | 0.6734 | +0.0160 | 0.6273 | 0.7267 |

Seven of ten cedar-rust images and seven of ten scab images improved in Dice.
Black-rot pooled Dice was effectively preserved, although individual black-rot
images moved in both directions. The checkpoint and training history are stored
in `artifacts/lesion_detection_real30_synthetic70`; full overlays and
per-image/per-disease metrics are stored in
`artifacts/lesion_evaluation_real30_synthetic70_all_images`.

### Interpretation and decision

The controlled result supports the synthetic-to-real imbalance hypothesis.
The 30/70 checkpoint is now the preferred general lesion model because it has
the best pooled test Dice and IoU, improves both weak diseases, and preserves
pooled black-rot performance.

Visual inspection still shows false positives on some stems, central veins,
leaf boundaries, and image edges, and difficult faded lesions remain imperfect.
The improvement therefore does not replace targeted structural-negative work.
The next experiment should retain 30/70 source sampling and introduce a smaller
or controlled hard-negative exposure so that structural false positives can be
reduced without repeating the earlier aggregate regression.

## 2026-10-05 — Controlled 5% hard-negative exposure

### Question

Can a small, explicitly controlled exposure to healthy hard negatives reduce
structural false positives without losing the improvement from 30/70
real-synthetic lesion sampling?

### Controlled change

- Extended source-balanced sampling to support a separate hard-negative
  probability and a fixed number of draws per epoch.
- Trained from `lesion_manifests/train.csv`, containing 30 real lesion pairs,
  300 synthetic lesion pairs, and 40 healthy hard negatives with empty masks.
- Retained a 30/70 real-synthetic ratio within lesion examples while reserving
  5% of total probability for hard negatives. The resulting overall probability
  mass was 28.5% real lesion, 66.5% synthetic lesion, and 5% hard negative.
- Fixed each epoch at 330 draws, matching the preceding 30/70 experiment rather
  than increasing the number of optimization steps to the 370-row manifest
  length.
- Kept the same paired augmentation, ResNet34 U-Net, Dice/focal loss, seed 42,
  256-pixel input size, batch size 8, 30 epochs, and real validation/test splits.
- Added a unit test verifying the three source groups receive exactly the
  requested probability mass. All five lesion tests passed before training.

### Results

The best checkpoint was restored from epoch 22. Validation Dice increased from
0.6901 at the default threshold to 0.6973 after calibrating the threshold to
0.70. On the fixed real test split, pooled Dice was 0.7009, IoU was 0.5395,
precision was 0.6317, and recall was 0.7871.

Compared with the preferred 30/70 checkpoint, Dice decreased by 0.0033, IoU by
0.0040, and precision by 0.0158, while recall increased by 0.0152. Mean
per-image Dice decreased from 0.7161 to 0.7107 and median per-image Dice from
0.7244 to 0.7020.

| Disease | 30/70 Dice | 5% hard-negative Dice | Change |
| --- | ---: | ---: | ---: |
| Black rot | 0.8317 | 0.8549 | +0.0232 |
| Cedar rust | 0.6657 | 0.6609 | -0.0048 |
| Scab | 0.6734 | 0.6621 | -0.0113 |

The checkpoint and history are stored in
`artifacts/lesion_detection_real30_synthetic70_hardneg05`. All 30 overlays,
pooled metrics, per-image metrics, and per-disease metrics are stored in
`artifacts/lesion_evaluation_real30_synthetic70_hardneg05_all_images`.

### Interpretation and decision

The 5% exposure improved black-rot Dice and overall recall, but it did not
improve pooled segmentation and reduced precision, cedar-rust Dice, and scab
Dice. Limited matched inspection of difficult cedar-rust and scab overlays did
not show an obvious structural false-positive reduction relative to the 30/70
checkpoint, consistent with the lower pooled precision.

The 30/70 checkpoint remains the preferred general lesion model. The 5%
hard-negative checkpoint is retained as controlled experimental evidence and
does not replace it. A follow-up should not simply raise the hard-negative
probability; it should first make hard-negative selection or loss contribution
more targeted, because both the earlier uncontrolled mixture and this controlled
5% exposure failed to improve the general test result.

### Comparison with the earlier augmented checkpoint

The 5% hard-negative checkpoint was also compared with the earlier checkpoint
that used paired augmentation but natural 9.1/90.9 real-synthetic sampling and
no hard negatives. On the same 30 real test images, pooled Dice increased from
0.6873 to 0.7009, IoU from 0.5236 to 0.5395, precision from 0.6281 to 0.6317,
and recall from 0.7589 to 0.7871. Mean per-image Dice increased from 0.6960 to
0.7107 and median per-image Dice from 0.6773 to 0.7020.

| Disease | Augmented Dice | 5% hard-negative Dice | Change |
| --- | ---: | ---: | ---: |
| Black rot | 0.8314 | 0.8549 | +0.0235 |
| Cedar rust | 0.6384 | 0.6609 | +0.0226 |
| Scab | 0.6574 | 0.6621 | +0.0047 |

Twenty of the 30 matched test images improved in Dice and ten declined. By
disease, six of ten black-rot, eight of ten cedar-rust, and six of ten scab
images improved. The largest gain was black rot image 10 (+0.1036 Dice); the
largest loss was cedar rust image 9 (-0.0823 Dice).

False-positive pixels decreased by 180 for black rot and 916 for cedar rust but
increased by 1,535 for scab, producing a net increase of 439 across the test
set. This agrees with the mixed precision result and shows that the comparison
does not establish a general structural false-positive improvement.

This comparison must not be interpreted as the effect of hard negatives alone:
the newer checkpoint differs from the augmented checkpoint in both 30/70
real-synthetic sampling and 5% hard-negative exposure. The controlled comparison
against the 30/70 checkpoint above isolates the hard-negative addition and shows
that it did not improve the preferred model.

## 2026-10-07 — Held-out healthy structural-negative benchmark

### Question

Does the controlled 5% hard-negative checkpoint reduce false alarms on unseen
healthy leaves enough to justify its small diseased-test Dice regression?

### Benchmark and controls

- Manually reviewed and fixed 50 healthy Plant Pathology 2020 images that were
  absent from lesion training and from the 40 selected hard negatives.
- Included multi-label structural coverage for stems, veins, folds, holes, edge
  damage, shadows, sunlight, image edges, and natural backgrounds.
- Verified all source labels as healthy and every generated lesion mask as
  exactly empty.
- Found no SHA-256 duplicates against the complete lesion training manifest or
  inside the benchmark.
- Found no near-duplicate at or below Hamming distance 4 using a 256-bit
  difference hash; the nearest observed distance was 91.
- Marked every manifest row evaluation-only and evaluated the three frozen
  checkpoints without retraining, using each checkpoint's stored 0.70 threshold.

### Results

| Checkpoint | Any predicted lesion | FP pixel rate | Mean predicted area | Components | Mean component size | Maximum component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Paired augmentation | 100% | 8.1856% | 5,364.52 px | 1,821 | 147.30 px | 5,001 px |
| Preferred 30/70 | 98% | 5.3483% | 3,505.06 px | 1,754 | 99.92 px | 2,675 px |
| 30/70 plus 5% hard negatives | 94% | 0.3578% | 234.48 px | 648 | 18.09 px | 310 px |

Against pure 30/70, the 5% checkpoint reduced false-positive pixel area by
93.3%, component count by 63.1%, and maximum component size by 88.4%. Its
false-positive pixel rate improved in every category, with relative reductions
from 89.5% for shadow images to 94.5% for vein images. The strict any-pixel rate
remained high because even one thresholded pixel counts as a false alarm.

Mean probabilities were 0.2526, 0.3513, and 0.4119 respectively, while maximum
probability reached approximately 1.0 for all three. The apparently higher mean
probability but much smaller thresholded area for the 5% model reflects a
different score distribution/calibration and does not contradict the binary
false-positive-area result.

### Decision

The healthy benchmark reveals a useful trade-off that the diseased test set
could not: a 0.0033 pooled-Dice loss is accompanied by a dramatic reduction in
healthy structural false-positive area. Investigate a controlled dual-objective
or separate healthy-batch loss next. Do not blindly increase hard-negative
exposure, and retain pure 30/70 as the general reference checkpoint during the
next controlled comparison.

The fixed benchmark is under `data/healthy_structural_benchmark`; detailed
evaluation tables are under
`artifacts/healthy_structural_benchmark_evaluation`; methodology and commands
are summarized in `HEALTHY_BENCHMARK.md`.

## Maintenance convention

For every future experiment, append a dated section containing:

- the question or hypothesis;
- the exact code or data change;
- controlled variables;
- metrics and artifact locations;
- interpretation and limitations;
- the decision made from the result.
