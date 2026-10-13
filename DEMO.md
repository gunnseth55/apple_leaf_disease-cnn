# End-to-end apple leaf demo

This local Gradio prototype illustrates the two-model pipeline. Field-image
classification remains poor, so this is a research demonstration rather than
reliable field diagnosis. The softmax score is uncalibrated and can be high on
incorrect predictions. Segmentation can highlight healthy structures.

## Run

From the repository root with the virtual environment activated:

```powershell
python -m pip install -e .
appleleaf-demo
```

Open **http://127.0.0.1:7860**. Upload an image, use the crop tool in the editor
to frame the main leaf, and apply the crop. Select **Analyze selected crop**.
Both models receive exactly that crop with their own original preprocessing.
Keep the whole leaf and all visible lesions inside the crop. Painting, erasing,
additional layers and editor resizing are disabled.

Use **Compare crop with full image** to view both sets of softmax scores. The
three result panels show the analyzed crop, binary mask and red overlay.
**Analyze full image** provides an uncropped baseline. Without applying a crop,
the editor initially contains the full image; analyzed dimensions are shown
so the input can be checked. A new upload resets the editor and clears results.
The server listens locally, with no public sharing link. Stop it with Ctrl+C.

Without reinstalling console commands:

```powershell
$env:PYTHONPATH = "src"
python -m appleleaf.cli.demo
```

## Frozen model configuration

| Task | Checkpoint | Input | Processing |
| --- | --- | --- | --- |
| Classification | `artifacts/efficientnet_b0/apple_efficientnet_b0_checkpoint.pt` | 224 × 224 | RGB, bicubic resize, ImageNet normalization, softmax |
| Default lesions | `artifacts/lesion_detection_real30_synthetic70/unet_resnet34_lesion_checkpoint.pt` | 256 × 256 | RGB, bilinear resize, ImageNet normalization, sigmoid, saved threshold 0.70 |
| Comparison lesions | `artifacts/lesion_detection_real30_synthetic70_hardneg05/unet_resnet34_lesion_checkpoint.pt` | 256 × 256 | Same processing, its saved threshold 0.70 |

The original 30/70 segmenter has the best reported diseased-image Dice of these
experiments (0.7043). The hard-negative comparison has Dice 0.7009 and substantially
fewer healthy-image false-positive pixels. See [HEALTHY_BENCHMARK.md](HEALTHY_BENCHMARK.md).

Both models run independently. A healthy classification does not suppress the
lesion mask. If the classifier predicts healthy but the segmenter selects pixels,
an explicit model-disagreement warning appears. Those pixels remain visible as
a separate segmentation output and must not be interpreted as a diagnosis.
If no pixels are selected, the interface still avoids claiming the leaf is
disease-free. Outputs include all four disease scores, the highest-score class and
softmax score, the analyzed leaf image, a red lesion overlay, and a binary mask.
The three image panels are always shown separately in that order: leaf image,
binary mask, red overlay. The overlay blends red at 45% opacity over the original
crop. No smoothing, component filtering or healthy-class suppression is applied.
The mask is thresholded on the 256 × 256 grid and resized to the analyzed image
dimensions with nearest-neighbor interpolation. Highlighted area describes the
model image grid, not a leaf-area disease severity estimate. Image orientation
is corrected using EXIF before either model sees it.

No training, threshold tuning, checkpoint writes, or pretrained-weight downloads
occur. Architecture, input size, class mapping, and saved thresholds are checked
at startup. Missing checkpoints produce an actionable startup error. Checkpoints
are local artifacts and are not included in the public repository.

## Options

```powershell
appleleaf-demo --device cpu --port 7861
appleleaf-demo --no-comparison
appleleaf-demo --classifier-checkpoint C:\path\classifier.pt --segmenter-checkpoint C:\path\segmenter.pt --no-comparison
```

The comparison checkpoint can be changed using `--comparison-checkpoint`.
The default `auto` device uses CUDA when available, otherwise CPU. The demo uses
the EfficientNet reference, regardless of the separate robustness experiment.

## Checks

```powershell
python -m unittest discover -s tests -p test_demo.py
```

These checks cover segmentation preprocessing parity with the existing evaluation
loader, threshold boundaries and overlay alignment, the classifier/segmenter
input sizes, and segmentation remaining visible for a healthy class prediction.

With the default demo server running and the local lesion test dataset present:

```powershell
python scripts/smoke_test_demo.py
```

This uploads a real test image through both Gradio API paths and checks score
normalization, original-image pixel parity, output dimensions, binary PNG masks,
and unchanged checkpoint hashes. It writes a report and overlays to
`artifacts/demo_verification/`. This is an integration check, not a new accuracy
benchmark. All 20 repository tests and both API paths passed during development;
browser UI inspection was unavailable in that session.

For the complete five-case workflow check, run:

```powershell
python scripts/test_demo_workflow.py
```

This starts a temporary local Gradio server on port 7861 and closes it after
testing. It uses the first held-out PlantVillage example by path for each class,
without filtering for correct predictions, plus a known difficult Kashmir scab
failure (its dataset label remains preliminary). Both segmenters are tested.
Forward hooks check the tensors received by the actual models against independent
224px bicubic and 256px bilinear ImageNet preprocessing. The check also verifies
exact original pixels, nearest-neighbor mask mapping, red overlay pixel alignment,
softmax scores, saved thresholds, conditional warnings and unchanged checkpoint
hashes. Results, masks, overlays and a contact sheet are saved under
`artifacts/demo_workflow_test/`. Alignment checks verify image coordinates, not
biological lesion accuracy; these five cases are not a new accuracy benchmark.

## Paired cropping experiment

```powershell
python scripts/test_demo_crops.py
```

This uses a temporary server on port 7862 to compare full images with fixed,
recorded manual crops through the actual editor initialization and comparison
API. It tests five dataset-labelled cases and, when present locally, the two
recent uploaded field-image screenshots. Each segmenter is tested. It checks
that both model tensors come from the same crop, exact preprocessing, cropped
image pixel preservation, binary-mask alignment, unchanged overlay regions and
checkpoint hashes. Results and contact sheet: `artifacts/demo_crop_test/`.

The seven-case run passed all 14 paired paths. The four PV labels remained
correct; the difficult preliminary Kashmir scab case remained predicted healthy.
One recent upload changed from scab (50.35%) to black rot (86.25%); the other
remained scab (42.26% to 59.62%). Those uploads have unverified disease labels,
so changed or higher scores do not establish an accuracy improvement. Crops were
fixed by leaf position, not optimized against prediction scores.

## Same-crop segmentation recheck

```powershell
python scripts/compare_demo_segmenters.py
```

Rechecks both frozen segmenters on the seven saved crops at their own stored
thresholds (both 0.70). Saves probability arrays, masks, overlays and two-row
comparison figures under `artifacts/demo_segmenter_recheck/`. Masks reproduced
the prior crop run exactly and checkpoint hashes were unchanged.

Keep 30/70 as the demo default and hardneg05 as a comparison. Although hardneg05
has a strong prior healthy-benchmark benefit, it did not consistently remove
background markings on these field crops. In one visually checked background
rectangle it selected 584 pixels versus 503 for 30/70. On the difficult Kashmir
crop it selected 43.02% of the grid versus 4.68%; the healthy PV crop had 0.42%
versus zero. These are case comparisons, not a new accuracy benchmark. Without
a complete leaf mask and lesion ground truth, total selected area must not be
called false-positive area. Every predicted region remains displayed.
