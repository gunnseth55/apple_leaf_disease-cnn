# EfficientNet-B0 robustness experiment

Prepared 2026-10-10. **No new training has been run.** The user runs all epochs.
The original EfficientNet-B0 is the **final in-distribution model candidate**, not
a final deployment model. Preserve its epoch-11 checkpoint unchanged. Interface
integration remains deferred until robustness and independent external evidence
are sufficient.

## What differs between the image domains

Measured 2,214 PlantVillage training images and the historical 260-image Kashmir
evaluation subset. Visually inspected 12 PlantVillage and nine Kashmir images
selected early/middle/late by sorted filename within each represented class,
plus the earlier 12-example Kashmir failure review. Selection is descriptive,
not representative sampling for estimating prevalence or a plant pathology review.

| Factor | PlantVillage observations | Kashmir observations | Implication and limits |
| --- | --- | --- | --- |
| Background complexity | Reviewed samples mainly show grey/purple, fine-textured surfaces and leaf shadows. | Fabric/stone-like textures and saturated pink/blue surfaces; some backgrounds are simple. | Background colour and texture differ; complexity is not uniformly greater. Train with independent background variation. |
| Lighting | Shadows and glossy highlights already occur. Whole-image luminance has a narrower middle range. | Pale underside views, dark glossy blades and uneven illumination; a wider middle luminance range. | Brightness, contrast, gamma and smooth shadows are plausible perturbations. Whole-image statistics include backgrounds. |
| Leaf scale and orientation | 256 x 256 square images; broad blades, stems and multiple orientations. | Mostly tall/narrow frames, varied framing and occasional cropping/oblique views. | Vary orientation and scale without removing lesions. Existing full-image resize distorts tall images; an aspect-preserving preprocessing ablation can follow separately. |
| Underside versus topside | Reviewed samples show varied vein visibility and surface texture; leaf side is not annotated. | Pale undersides and prominent veins are particularly apparent in reviewed healthy-folder images. | Rotation/flips cannot manufacture the opposite leaf surface. Reliable coverage needs independently sourced, correctly labeled real views. |
| Torn/damaged tissue | Some reviewed samples already have missing margins or damaged tips. | Prior failure review includes larger irregular holes, tears and brown margins. | Keep this as a possible confound; do not synthesize tears or call arbitrary damage a disease. |
| Disease severity | Small discrete marks through larger discoloration are visible. | Mildly marked, broadly brown and heterogeneous damaged blades occur. | No calibrated severity annotations permit a numerical severity comparison. Colour/blur augmentation does not create new disease stages. |
| Image quality | Standardized resolution; some softness and texture remain. | Variable original resolution, blur/soft focus and inconsistent detail. | Use moderate blur/JPEG perturbations. Resolution alone does not measure quality. |
| Lesion appearance | Reviewed scab samples have dark patches; black-rot samples include discrete tan/brown marks; rust samples show orange/brown marks. | Discrete marks, broad brown patches, yellowing and irregular damage; some scab-folder leaves predicted healthy visibly have brown patches. | Descriptive appearances do not establish diagnosis. Kashmir rot-to-black-rot remains unverified. |

Quantitative image audit:

| Measurement | PlantVillage train | Kashmir external |
| --- | ---: | ---: |
| Median native width / height | 256 / 256 | 353.5 / 714.5 |
| Median width-to-height ratio | 1.000 | 0.506 |
| Whole-image luminance mean, median | 116.22 | 124.29 |
| Whole-image luminance mean, 10th–90th percentiles | 94.07–135.54 | 90.65–161.38 |
| High-frequency variance at 128 px, 10th percentile | 309.50 | 67.43 |

High-frequency variance is confounded by veins, lesions, texture and compression;
it is not a standalone blur metric. No causal explanation is established by this
audit. CSV measurements and the sample manifest are in
`results/2026-10-10_robustness_plan/`. Four source-image contact sheets are in
`artifacts/domain_robustness_analysis/`.

## First controlled experiment: augmentation only

Hypothesis: a wider, label-preserving training distribution improves background
stress robustness and external scab/healthy performance while retaining strong
PlantVillage performance. This remains a hypothesis until the user trains and
evaluates the candidate.

Keep the architecture, ImageNet-1K V1 initialization, four-class mapping, seed 42,
train/validation/test split, 224 x 224 deterministic evaluation preprocessing,
loss, optimizer, batch size and 15-epoch schedule identical to the frozen model's
experiment. Start from ImageNet weights, not by continuing training the selected
PlantVillage checkpoint. The experimental variable is the training augmentation
profile. Select the best checkpoint by **original PlantVillage validation
macro-F1**, with earliest ties; do not select by Kashmir or the test stress suite.

The robust profile uses only PlantVillage training rows:

- Keep original backgrounds with probability 0.35. For replacements, draw with
  equal probability a nonleaf crop from an independently sampled training image
  or a synthetic neutral/earth/foliage-coloured multiscale texture. Reject donor
  crops containing mask foreground; after eight unsuccessful attempts, use
  synthetic texture. No validation/test/Kashmir donor pixels enter training.
- Preserve foreground pixels exactly during background compositing. PlantVillage
  masks are imperfect, so this guarantee concerns mask foreground, not expert
  lesion boundaries. Donor patches and synthetic textures do not supply actual
  orchard scenes or new leaf anatomy.
- Horizontal/vertical flips. With probability 0.8, rotate up to 180 degrees on an
  expanded canvas, fit the entire image at scale 0.75–1.0, and vary its position.
  With probability 0.2, keep geometry apart from flips. Do not crop away lesions.
- With probability 0.8, jitter brightness/contrast by up to 0.30, saturation by
  0.15 and hue by 0.015. Use occasional gamma 0.75–1.30 and a smooth illumination
  gradient with darkest multiplier 0.70–0.90.
- Occasional moderate Gaussian blur (radius 0.3–1.0 at 224 px) and JPEG quality
  60–95. No lesion erasing, artificial tears, new pathology or relabeling.

Visual preview:
[augmentation_preview.jpg](artifacts/domain_robustness_analysis/augmentation_preview.jpg).
These perturbations approximate selected nuisance factors; they do not prove
real-world robustness or cover missing underside/severity/anatomical variation.

Mixed-source training is reserved for a **separate later experiment**. First
verify an independent non-Kashmir source's labels, class equivalence, provenance,
acquisition groups and exact/near-duplicate overlap with all validation/test sets.
Then define source-balanced sampling, a fixed budget and a separate validation
protocol before training. Do not silently import the existing Kashmir data,
lesion images, or held-out healthy benchmark. Source mixing is not enabled in
this first experiment, isolating the effect of augmentation.

## Training: run this yourself

From the project root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m appleleaf.cli.train_efficientnet `
  --dataset-path C:/Users/gunn/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3 `
  --augmentation-profile robust `
  --artifacts-dir artifacts/efficientnet_b0_robust `
  --epochs 15 --warmup-epochs 3 --batch-size 16 --device cpu
```

Schedule remains three head-only epochs (LR 0.001) and 12 full fine-tuning epochs
(backbone LR 0.0001, head LR 0.0003, cosine decay); AdamW weight decay 0.0001.
Evaluation normalization remains ImageNet RGB mean `(0.485, 0.456, 0.406)` and
std `(0.229, 0.224, 0.225)`.

The new checkpoint will be
`artifacts/efficientnet_b0_robust/apple_efficientnet_b0_robust_checkpoint.pt`.
The CLI refuses the frozen reference output folder and any nonempty run folder.
`--dry-run` checks data/configuration without downloading weights or running
epochs. The default `mild` profile remains the original training recipe.

## Paired comparison: run after training finishes

```powershell
./scripts/run_robustness_comparison.ps1
```

This runs inference only, never training, for both frozen reference and robust
candidate. The comparison checks matched image manifests and checkpoint hashes:

1. **PlantVillage test:** all 475 original test images; macro-F1, precision,
   recall, per-class F1, confusion matrix and confidence distribution.
2. **Background perturbations on the same test images:** original, white, blue,
   black, background-only and fixed multiscale neutral/earth/foliage texture.
   Texture is deterministic per image and identical for both models. Foreground
   is preserved for replacement tests; background-only deliberately removes it.
   These are controlled stress tests, not field accuracy estimates.
   Background-only is a shortcut diagnostic, not a normal classification task;
   its score can reflect scene or silhouette cues and must not be ranked like
   leaf-preserving conditions.
3. **Kashmir scab/healthy:** the fixed historical 160-image subset (117 scab,
   43 healthy), leaving the rot mapping out of scoring. Keep all four prediction
   classes: black-rot/cedar-rust outputs remain errors. No binary renormalization,
   tuning, labels, data files or exclusion decisions are changed. Original source
   label limitations and later visual conflicts still apply, so this remains
   preliminary. The frozen model's previously saved subset accuracy is 41.25%.
4. **Later verified black-rot/cedar-rust external sets:** pending appropriate
   independently verified, genuinely held-out images. No results are claimed.

The runner writes `artifacts/robustness_comparison/`, including per-model reports,
`comparison.csv` and `robust_minus_reference.csv`. To repeat, use a fresh folder:

```powershell
./scripts/run_robustness_comparison.ps1 -OutputRoot artifacts/robustness_comparison_run2
```

Do not automatically promote whichever candidate wins on Kashmir. Review the
PlantVillage tradeoff, background sensitivity, class-wise external recall and
confidence errors. Keep the reference frozen and avoid successive augmentation
tuning driven by this same external set. Kashmir has already been inspected, so
it is evaluation-only but no longer a blind benchmark; future unseen external
data is needed for an independent confirmation of any improvement.

## Verification performed by Codex

Eleven classification/data tests passed without training epochs. Checks cover
mask-foreground preservation, deterministic perturbations, finite 224 x 224
inputs, reference-folder protection, four-way scab/healthy scoring, checkpoint
roundtrip inference on synthetic tensors and legacy compatibility. The robust
dry run confirmed 2,214/475/475 split counts and mask availability; no model
weights were downloaded. Source and augmentation contact sheets were visually
inspected. Python compilation and PowerShell parser checks passed. The comparison
runner has **not** been executed, and the robust checkpoint does **not** yet exist.
