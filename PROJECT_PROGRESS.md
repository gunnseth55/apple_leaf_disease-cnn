# Project Progress Log

## 2026-10-08 — PlantVillage reproduction and Kashmir label audit

Both saved disease checkpoints exactly reproduced the original 475-image
PlantVillage test predictions and probabilities: baseline 90.32% accuracy,
macro-F1 0.911271; background augmented 83.79%, macro-F1 0.795268. Their hashes
match the checkpoints used for Kashmir. See
[PLANTVILLAGE_REPRODUCTION.md](PLANTVILLAGE_REPRODUCTION.md).

Recovered all 47 Kashmir Git LFS objects from a public mirror, validating their
declared size, SHA-256, and RGB readability. Saved a separate recovered dataset;
the original files and historical evaluation remain intact. Screened all 313
images in the three represented classes and visually reviewed 84 cross-folder
perceptual-duplicate candidates. Eight additional rot/scab conflict groups were
found beyond the three exact rot/healthy conflicts. Conservative exclusion of
25 conflicting rows and five same-class duplicate copies leaves 283 candidate
images (91 rot, 43 healthy, 149 scab), with disease labels still unverified.

The rot-to-black-rot mapping could not be confirmed: the mirror maps the names
in software, but the inspected documentation does not provide supporting
pathogen verification. The source Drive also contains leaf blotch, absent from
our local subset. Independent disease-label review remains necessary; no new
definitive Kashmir scores were claimed and integration remains paused. See
[KASHMIR_LABEL_AUDIT.md](KASHMIR_LABEL_AUDIT.md) and
`results/2026-10-08_kashmir_label_audit/` for evidence and review decisions.

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

## 2026-10-07 — Separate healthy-batch objective (dualhealthy05)

### Question and controlled change

Can a separately weighted healthy empty-mask loss reduce structural false
positives while preserving all 330 lesion draws and the 30/70 real-synthetic
lesion sampling of the reference experiment?

- Added `--healthy-manifest` and `--healthy-loss-weight` to the lesion training
  CLI and configuration. The auxiliary loader selects only `source=hard_negative`
  rows from `lesion_manifests/train.csv`: the original 40 training negatives.
- Trained lesion batches from `train_before_hard_negatives.csv`, retaining
  30/70 source sampling with replacement and 330 lesion draws per epoch.
- Each lesion batch receives one separate healthy batch, cycling the shuffled
  healthy loader when exhausted. One optimizer step combines lesion Dice/focal
  gradients with `0.05 * healthy BCE` gradients. This is a loss weight, not a
  5% healthy sampling probability; auxiliary batches add computation and healthy
  exposure without replacing lesion draws.
- Preserved the pretrained ResNet34 U-Net, paired augmentation, seed 42,
  batch size 8, input size 256, 30 epochs, AdamW settings, and real validation
  and test splits. Auxiliary training also updates batch-normalization statistics
  and consumes augmentation randomness; this is not an identical lesion-batch
  trajectory with only a scalar loss added.
- The 50 held-out healthy benchmark images were used only for frozen-checkpoint
  evaluation, not training or threshold calibration.

### Diseased results

Restored epoch 18, selected with validation Dice 0.6856 at threshold 0.50.
Validation calibration selected threshold 0.55 with Dice 0.6878 (the calibrated
value is from the reported terminal output). Checkpoint test loss is 0.4345.
The 30 real test images yielded pooled Dice 0.6933, IoU 0.5306, precision
0.6530, and recall 0.7390. Mean per-image Dice was 0.7029, median 0.7095,
and standard deviation 0.1108.

| Disease | Images | Dice | IoU | Precision | Recall | Mean image Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Black rot | 10 | 0.8295 | 0.7086 | 0.8123 | 0.8474 | 0.7963 |
| Cedar rust | 10 | 0.6395 | 0.4701 | 0.6379 | 0.6412 | 0.6432 |
| Scab | 10 | 0.6645 | 0.4975 | 0.5996 | 0.7450 | 0.6691 |

The five lowest image Dice scores were cedar_rust_2.jpg (0.4795),
cedar_rust_8.jpg (0.5061), cedar_rust_1.jpg (0.5535), scab_4.png (0.5591),
and cedar_rust_7.jpg (0.5889). Complete paths and scores are in the saved
per-image CSV.

### Healthy benchmark comparison

All four frozen checkpoints were evaluated on the same 50 healthy images using
their stored validation-calibrated thresholds.

| Checkpoint | Threshold | Diseased Dice | Healthy images with any prediction | Healthy FP pixel rate | Mean FP area (px) | FP components | Mean component (px) | Max component (px) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paired augmentation | 0.70 | 0.6873 | 50/50 (100%) | 8.1856% | 5,364.52 | 1,821 | 147.30 | 5,001 |
| Pure 30/70 | 0.70 | 0.7043 | 49/50 (98%) | 5.3483% | 3,505.06 | 1,754 | 99.92 | 2,675 |
| 30/70 + hardneg05 | 0.70 | 0.7009 | 47/50 (94%) | 0.3578% | 234.48 | 648 | 18.09 | 310 |
| 30/70 + dualhealthy05 | 0.55 | 0.6933 | 49/50 (98%) | 0.7709% | 505.22 | 390 | 64.77 | 881 |

Dualhealthy05 reduced healthy FP area by 85.6% and component count by 77.8%
relative to pure 30/70. Compared with hardneg05, it produced 39.8% fewer
components but 2.15 times the FP area, with larger components. Full probability,
category, and per-image statistics are preserved in the result snapshot.

### Interpretation, limitations, and decision

The auxiliary objective did not improve the combined result: diseased Dice is
about 0.011 below pure 30/70 and 0.0076 below hardneg05 using the displayed
scores, while healthy FP area is worse than hardneg05. Cedar rust remains
the weakest disease. Fewer components do not imply less false-positive area.
These are comparisons at each model's selected operating point, not a common
threshold comparison. The strict any-pixel healthy false-alarm rate remains
high. Small test sets and repeated comparisons do not establish statistical
significance or field performance.

Retain pure 30/70 as the general lesion reference and hardneg05 as the
healthy-FP-area reference. Preserve dualhealthy05 as experimental evidence;
do not promote it over either reference. Healthy benchmark checking is paused
for now as requested; no additional training, threshold tuning, or benchmark
runs are part of this recording step.

### Artifacts and verification

Original local outputs remain in:

- `artifacts/lesion_detection_real30_synthetic70_dualhealthy05/`
- `artifacts/lesion_evaluation_real30_synthetic70_dualhealthy05/`
- `artifacts/healthy_structural_benchmark_evaluation_dualhealthy05/`

Git-tracked CSV/JSON snapshots, checkpoint metadata, and reproduction commands
are under `results/2026-10-07_dualhealthy05/`. Checkpoint weights and overlay
PNGs remain local under the existing ignore policy. All six lesion unit tests
passed, including the auxiliary nonempty-mask rejection test, before committing.

## 2026-10-08 — Kashmir external classifier evaluation

Question: do the frozen baseline and background-augmented classifiers generalize
to the locally supplied Kashmir images before integration with lesion segmentation?

Added `appleleaf-evaluate-kashmir` with checkpoint class-order validation,
deterministic folder manifest, byte-level duplicate/conflict checks, explicit
exclusions, four-way predictions, present-class and four-class macro-F1,
per-class precision/recall/F1, confusion matrices, confidence statistics and bins,
all probabilities, and highest-confidence failure examples. No training or tuning
was performed; checkpoint RGB resize/ToTensor preprocessing was retained.

Of 313 image-named files, 47 are missing Git LFS image contents and six rows
represent three images with conflicting rot/healthy labels. Explicit exclusions
leave 260 images: 117 scab, 100 rot, 43 healthy. Rot-to-black-rot mapping remains
provisional. No cedar-rust ground truth exists; headline macro-F1 averages the
three represented classes while keeping all four prediction outputs.

Both classifiers achieved 63/260 correct (24.23% accuracy), below the 45.00%
majority-class reference. Baseline present-class macro-F1 was 0.2851;
background-augmented macro-F1 was 0.2495. Baseline made 64 errors among 78
predictions with confidence ≥0.90; augmented made 24 among 28. Excluding rot
entirely, scab/healthy accuracy was 28.75% and 35.00%, respectively.

Decision: neither classifier is validated for reliable external disease labeling.
Preserve the results without selecting/tuning a model on this benchmark. Resolve
missing files and the rot diagnosis for a definitive Kashmir result. Proceed next
to an independent externally annotated lesion benchmark, then demo integration.
Classification folders alone cannot provide segmentation ground truth.

Full report: `KASHMIR_EVALUATION.md`. Original outputs:
`artifacts/kashmir_evaluation/{baseline,background_augmented}/`. CSV/JSON/text
snapshots: `results/2026-10-08_kashmir/`. Three focused evaluation tests passed;
both full available-subset inference runs completed and the augmented failure
sheet was visually inspected. Training overlap and near duplicates remain unaudited.

## Maintenance convention

For every future experiment, append a dated section containing:

- the question or hypothesis;
- the exact code or data change;
- controlled variables;
- metrics and artifact locations;
- interpretation and limitations;
- the decision made from the result.

## 2026-10-10 — EfficientNet-B0 transfer-learning code prepared

Question: can an ImageNet-pretrained classifier improve on the small CNN while
retaining the existing PlantVillage split? The user selected EfficientNet-B0 and
explicitly reserved all training epochs for themselves.

Added `appleleaf.cli.train_efficientnet` and the
`appleleaf-train-efficientnet` console entry point. The default plan is 3 epochs
of classifier-head warmup with frozen backbone statistics, followed by 12 epochs
of full fine-tuning with separate backbone/head learning rates and cosine decay.
It uses ImageNet normalization, full-image bicubic resize to 224 x 224, mild
training-only geometry/colour augmentation, training-only class weights, and
validation macro-F1 checkpoint selection. No Kashmir or lesion data are used.

The decoded-image deduplication, seed-42 stratified split, and four-class mapping
remain unchanged. The run saves split manifests, configuration, per-epoch best
weights and history, then final test metrics, probabilities, curves and confusion
matrix. Existing output folders cannot be overwritten. Architecture/preprocessing
metadata now drive checkpoint loading, evaluation, background audits and Grad-CAM;
legacy CNN checkpoints preserve their original preprocessing.

Verification: four transfer-learning unit tests and three existing Kashmir
evaluation tests passed. These checked configuration, preprocessing, head/backbone
gradient flags, EfficientNet inference/checkpoint roundtrip, legacy checkpoint
loading, and explicit output-folder rejection. An EfficientNet Grad-CAM smoke
check produced four probabilities and a 224 x 224 map. A dry run on cached
PlantVillage version 3 verified readability/deduplication and confirmed counts
of 2,214 train, 475 validation and 475 test. No model download, optimizer step,
training epoch, or benchmark inference on actual dataset images was performed.

There are no new accuracy or macro-F1 results. This changes more than architecture
(pretraining, resolution, augmentation and optimizer also change), so it is a
transfer-learning experiment rather than an isolated architecture comparison.
External generalization remains unproven. The user will run training and supply
the resulting metrics before selecting a model or proceeding with integration.

Commands and exact settings: `EFFICIENTNET_TRANSFER.md`. Intended local outputs:
`artifacts/efficientnet_b0/`. Training remains **not started**.

## 2026-10-10 — User-completed EfficientNet-B0 training results

The user subsequently ran all 15 epochs on CPU: 3 frozen-backbone head epochs
and 12 fine-tuning epochs. Codex did not run training. The saved configuration
retains the original PlantVillage-only seed-42 split and transfer-learning
protocol described above; no external images entered training or selection.

The best checkpoint was epoch 11, selected by validation macro-F1 1.0000.
Test accuracy was 0.9978947368 (**474/475, 99.79%**) and macro-F1 was
**0.9981701493**. Per-class F1: scab 0.9947 (95 images), black rot 1.0000
(93), cedar rust 1.0000 (41), healthy 0.9980 (246). The sole error was scab
predicted healthy: `fb942296-ba33-40e5-ada0-6700365cf71d___FREC_Scab 2919.JPG`.
Compared with the original CNN's 429/475 (90.32%), this is 45 additional correct
predictions and a 9.47-percentage-point accuracy increase on the same test split.

Verification recalculated accuracy and macro-F1 from saved NPZ predictions and
matched `metrics.json` exactly. All 475 labels/predictions match the saved CSV.
No model inference, optimizer steps or training epochs were run for recording.
Metrics, configuration and history snapshots are saved under
`results/2026-10-10_efficientnet_b0/`; original weights, manifests, predictions
and figures remain under `artifacts/efficientnet_b0/`. Full per-class results
and reproduction instructions are in `EFFICIENTNET_TRANSFER.md`.

Decision: retain this frozen checkpoint as the stronger PlantVillage classifier
candidate. The high score establishes performance on this split, not field
reliability. Multiple training choices differ from the CNN reference and near
duplicates/related views have not been exhaustively audited. EfficientNet's
external evaluation remains pending; Kashmir labels remain unresolved. Do not
train or tune on the external evaluation set.

### Best-checkpoint preservation and metadata

Verified the saved `artifacts/efficientnet_b0/apple_efficientnet_b0_checkpoint.pt`
against training history: selected epoch 11 is the earliest maximum validation
macro-F1 (1.0000). The best checkpoint already exists and was retained unchanged.
Recorded RGB bicubic 224 x 224 preprocessing, no crop, ImageNet normalization
mean/std, AdamW, warmup/fine-tuning learning rates and cosine schedule, 15 completed
epochs (3 + 12), class mapping and checkpoint name in `EFFICIENTNET_TRANSFER.md`.
Full metadata and checkpoint SHA-256 are saved in
`results/2026-10-10_efficientnet_b0/checkpoint_metadata.json`, with a local copy
alongside the weights. Future checkpoints also explicitly embed normalization
values and checkpoint name. No training or model inference was run for this step.

### Evaluation reporting prepared

Expanded the PlantVillage evaluation CLI to save macro-F1, per-class
precision/recall/F1/support, CSV/PNG confusion matrices, confidence summaries,
ten confidence bins, a correct/error confidence histogram and per-image
probabilities. Class-wise recall is the recall column in the per-class table.
Evaluation records checkpoint SHA-256 and preprocessing, validates class mapping,
and refuses nonempty output directories. CLI help, Python compilation and diff
checks passed. No training or evaluation inference was run by Codex; evaluation
commands and output descriptions are in `EFFICIENTNET_TRANSFER.md`.

## 2026-10-10 — User-run EfficientNet PlantVillage and Kashmir evaluations

Question: does the improved PlantVillage classifier generalize externally?
The user ran both evaluation commands with frozen epoch-11 weights, saved
preprocessing and no training/tuning. PlantVillage reproduced 474/475 correct
(99.79%), macro-F1 0.9981701493. Saved ordered labels/predictions match the
original test output exactly; maximum probability difference is 2.2352e-08.

Kashmir: 68/260 correct (26.15%), present-class macro-F1 0.2248473493, all-four
macro-F1 0.1686355120. Per-class recall is scab 27.35%, provisional rot 2.00%,
healthy 79.07%; cedar rust has no ground-truth support. This is five additional
correct predictions versus either CNN, but lower present-class macro-F1 and
accuracy below the 45.00% majority reference. The model predicts healthy for
136/217 diseased-folder images. Excluding rot, accuracy is 66/160 (41.25%).

Confidence is uncalibrated: mean 0.8440; errors 0.8534 versus correct 0.8175.
Of 141 predictions at confidence >=0.90, 107 are wrong (75.89%). The saved
inventory matches the original CNN's 260-image manifest exactly and excludes
47 LFS pointers and six exact-conflict rows. Later visual conflicts and
unverified diagnoses remain; this is not the recovered candidate inventory.

Both evaluations record the preserved checkpoint hash. Codex checked saved
outputs only, with no inference or training. Original outputs:
`artifacts/efficientnet_b0_evaluation_full/` and
`artifacts/kashmir_evaluation/efficientnet_b0/`. Tracked report snapshots:
`results/2026-10-10_efficientnet_b0/{plantvillage_evaluation,kashmir}/`.
Full tables are in `EFFICIENTNET_TRANSFER.md` and `KASHMIR_EVALUATION.md`.

Decision: retain EfficientNet as the PlantVillage classifier reference, without
claiming external diagnostic reliability. Distribution mismatch and label
problems are possible contributors, not proven explanations. Do not tune on
Kashmir; independent label verification and external lesion evaluation remain
the next evidence needed before integration.

## 2026-10-10 — Frozen classifier class breakdown and visual failure review

At the user's request, retained EfficientNet-B0 as the current best classifier
based on PlantVillage validation macro-F1. Verified the checkpoint SHA-256
before and after review: unchanged. No training, inference, threshold selection,
label changes or evaluation exclusions were performed.

Added `scripts/review_kashmir_efficientnet.py` to analyze saved predictions and
generate reproducible class tables and two contact sheets. Scab recall is
32/117 (27.35%); healthy recall 34/43 (79.07%). The rot folder produces 54
healthy, 44 scab, two black-rot and zero cedar-rust predictions. Cedar rust is
predicted only once overall (1/260, 0.38%), on scab-folder K266 at confidence
0.3332. There is no true cedar-rust support, so its recall is unmeasured.
Healthy precision is 20.00% despite its higher recall. Joining the previous
label audit identified 16 known visual-conflict rows still in this historical
260-image subset; scores and exclusions were preserved.

Visually inspected 12 selected photos: 11 disagreements and one rot/black-rot
agreement for context. Selection covers highest/middle/selected lowest-confidence
examples across major prediction patterns. Observed brown/tan patches in some
scab-to-healthy errors, dark pink-background blades in rot-to-healthy examples,
torn/holed blades in rot-to-scab examples, and pale underside views in some
healthy-to-scab examples. These observations do not establish diagnoses or
causes. One selected example (K051) is in the existing conflict group G09.
Individual observations and selection provenance were recorded after inspection.

Full report: `KASHMIR_EFFICIENTNET_REVIEW.md`. Visual sheets:
`artifacts/kashmir_evaluation/efficientnet_b0_review/`. Tracked CSV/JSON review
outputs: `results/2026-10-10_efficientnet_b0/kashmir_review/`.

Decision: keep the classifier frozen, keep overall Kashmir accuracy 26.15% and
present-class macro-F1 0.2248 explicitly preliminary until rot/source labels are
verified, and do not treat visual agreement as an independent diagnostic label.
