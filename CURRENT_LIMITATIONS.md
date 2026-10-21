# Current project limitations

Status: 2026-10-21, based on the recorded experiments through 2026-10-18.
This document summarizes existing evidence; no new training or evaluation was
performed to prepare it.

## Classification and field generalization

- The frozen PlantVillage-only EfficientNet-B0 reference achieved 99.79%
  accuracy (474/475) and 0.9982 macro-F1 on the PlantVillage test split. This
  establishes performance on that split, not reliable orchard diagnosis.
- On the historical 260-image Kashmir subset, the same model achieved only
  26.15% accuracy and 0.2248 present-class macro-F1. These scores are preliminary
  because the source labels have unresolved conflicts and the mapping from
  "Apple Rot" to black rot is unconfirmed.
- Confidence is uncalibrated. In that Kashmir evaluation, 107 of 141 predictions
  with softmax scores of at least 0.90 were wrong. A high score cannot be treated
  as a verified probability of a correct diagnosis.
- The classifier has four outputs: scab, black rot, cedar rust, and healthy.
  These classes do not cover every disease or source of leaf damage, and the
  current demo has no unknown-class rejection output.

Sources: [EfficientNet results](EFFICIENTNET_TRANSFER.md),
[Kashmir evaluation](KASHMIR_EVALUATION.md), [demo](DEMO.md).

## Mixed-source candidate is not yet test-validated

- C1 fine-tuning on PlantVillage and PP2020 completed ten epochs. The selected
  checkpoint reached 1.0000 PlantVillage validation macro-F1 and 0.9508 PP2020
  present-three-class validation macro-F1. Validation was used to select the
  checkpoint, so these are not independent test results.
- Neither frozen test split has been evaluated for C1 in the recorded results.
  Its improvement over the reference on held-out field tests remains unknown.
- PP2020 provides scab, rust, and healthy examples, but no black-rot field class.
  Black-rot supervision still comes from PlantVillage; orchard black-rot
  reliability remains unestablished.
- C1 has not replaced the frozen classifier used by the demo. The separate
  PlantVillage-only robustness augmentation experiment is prepared but has no
  recorded completed training run or robustness comparison.

Sources: [C1 training](FIELD_CLASSIFIER_TRAINING.md),
[robustness experiment](ROBUSTNESS_EXPERIMENT.md), [demo configuration](DEMO.md).

## Dataset labels, independence, and evaluation coverage

- All 47 missing Kashmir images were recovered. Recovery solved file availability,
  but did not verify diagnoses. The audit identified 25 conflicting rows and five
  same-label duplicate copies; the remaining 283 images are inventory candidates,
  not a validated diagnostic benchmark.
- Historical Kashmir scores still describe the original 260-image evaluation;
  they are not results for the recovered or audited inventory. Kashmir contains
  no cedar-rust ground truth, so it cannot measure cedar-rust recall.
- Acquisition and leaf IDs are unavailable for Kashmir. Related views and
  transformed duplicates may remain; PlantVillage related-view/near-duplicate
  independence has not been exhaustively established either.
- PP2020 split-v1 enforces audited duplicate groups and previous-use restrictions.
  Its Cedar prior-use review is bounded by available local records and cannot
  establish the absence of deleted or unrecorded uses.
- Kashmir and the healthy benchmark have already been inspected. Further
  comparisons on these sets cannot substitute for confirmation on genuinely
  unseen external data, and test results must not guide tuning.

Sources: [Kashmir label audit](KASHMIR_LABEL_AUDIT.md),
[EfficientNet results](EFFICIENTNET_TRANSFER.md),
[PP2020 split protocol](PP2020_FIELD_SPLITS.md),
[Cedar usage review](PP2020_CEDAR_USAGE_REVIEW.md),
[held-out preservation](PP2020_HELDOUT_PRESERVATION.md).

## Lesion segmentation

- The annotated real-image dataset is small: 30 training, 15 validation, and 30
  test images, with only ten test examples per disease. Synthetic images add
  training examples but do not establish generalization to real orchard images.
- The retained 30/70 segmenter has reported diseased-test Dice of 0.7043; the
  hardneg05 comparison has 0.7009. Cedar rust remains the weakest disease in the
  recorded comparison. Independent externally annotated lesion evaluation is
  still needed.
- Healthy structures and background regions can be highlighted. On the 50-image
  healthy benchmark, the default segmenter selected at least one pixel in 98%
  of images, with a 5.3483% false-positive pixel rate. Hardneg05 reduced that
  pixel rate to 0.3578%, but still selected pixels in 94% of images.
- The hard-negative benefit did not consistently remove background markings on
  the inspected field crops. Those crop comparisons have no complete lesion
  ground truth and are not an accuracy benchmark.
- Thresholds affect the masks. Comparisons involving dualhealthy05 use different
  saved operating thresholds (0.55 versus 0.70), and the small test sets and
  repeated comparisons do not establish statistical significance.
- Masks are binary lesion predictions, not disease-specific pixel diagnoses.
  Highlighted image-grid area is not a validated percentage of diseased leaf
  area or a disease-severity measurement. Whole-leaf masks and Grad-CAM maps
  cannot substitute for lesion annotations.

Sources: [lesion pipeline](LESION_DETECTION.md),
[healthy benchmark](HEALTHY_BENCHMARK.md), [demo recheck](DEMO.md),
[experiment log](PROJECT_PROGRESS.md).

## Demo and reproducibility

- Classification and segmentation run independently. A healthy prediction can
  coexist with a nonempty lesion mask; the disagreement warning does not resolve
  which model is correct. An empty mask does not establish a disease-free leaf.
- Both models resize the selected image or crop (224 x 224 for classification,
  256 x 256 for segmentation). Cropping changes the input and can change the
  prediction; the small crop experiment did not establish improved accuracy.
- Workflow checks verify preprocessing, image alignment, thresholds, and artifact
  preservation. Passing them does not validate biological diagnoses or field
  lesion accuracy.
- The demo requires local checkpoints, which are not included in the public
  repository. Datasets and lesion manifests must also be supplied locally;
  a repository checkout alone is insufficient to reproduce every experiment.
- The current deliverable is a local research prototype. Reliable field diagnosis
  and a validated combined classification/segmentation pipeline remain unproven.

Sources: [demo setup and checks](DEMO.md), [lesion data contract](LESION_DETECTION.md).

## Evidence still needed

1. A source-wise comparison of C1 and the frozen reference on the preserved
   PlantVillage and PP2020 test splits, without test-driven tuning.
2. Independently verified external disease labels, including field black rot,
   and genuinely unseen data for confirmation.
3. An independent orchard lesion benchmark with pixel-level annotations and
   healthy examples to measure both missed lesions and false positives.
4. Evidence for confidence calibration, unknown-input handling, and combined
   pipeline reliability before making field-diagnostic claims.
