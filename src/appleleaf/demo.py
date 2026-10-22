"""Frozen-checkpoint inference for the local pipeline demonstration."""

from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.transforms import functional

from appleleaf.config import APPLE_CLASSES
from appleleaf.datasets import checkpoint_transform
from appleleaf.engine import choose_device
from appleleaf.workflow import load_model as load_classifier
from appleleaf_lesion.workflow import load_model as load_segmenter


CLASSIFIER_CHECKPOINTS = {
    "C1 field-adapted (PlantVillage + orchard)": Path("artifacts/efficientnet_b0_field_c1_v1/apple_efficientnet_b0_field_c1_checkpoint.pt"),
    "Reference (PlantVillage)": Path("artifacts/efficientnet_b0/apple_efficientnet_b0_checkpoint.pt"),
    "Robust (PlantVillage + augmentation)": Path("artifacts/efficientnet_b0_robust/apple_efficientnet_b0_robust_checkpoint.pt"),
}
DEFAULT_CLASSIFIER = next(iter(CLASSIFIER_CHECKPOINTS.values()))
DEFAULT_SEGMENTER = Path("artifacts/lesion_detection_real30_synthetic70/unet_resnet34_lesion_checkpoint.pt")
DEFAULT_COMPARISON = Path("artifacts/lesion_detection_real30_synthetic70_hardneg05/unet_resnet34_lesion_checkpoint.pt")
PRIMARY = "30/70 real/synthetic (reference)"
COMPARISON = "30/70 + 5% hard negatives (comparison)"


def prepare_image(image):
    if image is None:
        raise ValueError("Upload an apple leaf image first.")
    if not isinstance(image, Image.Image):
        raise ValueError("Please upload a readable image.")
    return ImageOps.exif_transpose(image).convert("RGB")


def prepare_crop(editor):
    """Use the applied editor composite, never its uncropped background."""
    if not isinstance(editor, dict) or editor.get("composite") is None:
        raise ValueError("Upload an image, select a crop, and apply it in the crop editor first.")
    return prepare_image(editor["composite"])


def seed_crop(image):
    if image is None:
        return None
    image = prepare_image(image)
    return {"background": image, "layers": [], "composite": image}


def segmentation_input(image, image_size=256):
    # Match LesionDataset evaluation: bilinear square resize and ImageNet normalization.
    image = functional.resize(image, [image_size, image_size])
    return functional.normalize(
        functional.to_tensor(image),
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )


def lesion_overlay(image, probability, threshold):
    """Threshold at model resolution, then map the binary mask to the source image."""
    mask = Image.fromarray((probability >= threshold).astype(np.uint8) * 255)
    mask = mask.resize(image.size, Image.Resampling.NEAREST)
    selected = np.asarray(mask) > 0
    pixels = np.array(image)
    pixels[selected] = np.rint(
        0.55 * pixels[selected] + 0.45 * np.array([255, 0, 0])
    ).astype(np.uint8)
    return Image.fromarray(pixels), mask


class DemoPipeline:
    def __init__(self, classifier=DEFAULT_CLASSIFIER, segmenter=DEFAULT_SEGMENTER,
                 comparison=DEFAULT_COMPARISON, device="auto"):
        self.device = choose_device(device)
        self.paths = {PRIMARY: Path(segmenter)}
        if comparison is not None:
            self.paths[COMPARISON] = Path(comparison)
        for path in [Path(classifier), *self.paths.values()]:
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")
        self.classifier, metadata = load_classifier(classifier, self.device)
        if (metadata.get("architecture") != "efficientnet_b0"
                or metadata.get("image_size") != 224
                or metadata.get("preprocessing") != "imagenet_rgb_resize"):
            raise ValueError("Classifier must be an EfficientNet-B0 224px ImageNet-normalized checkpoint.")
        mapping = metadata["class_to_index"]
        if set(mapping) != set(APPLE_CLASSES) or sorted(mapping.values()) != list(range(4)):
            raise ValueError("Classifier checkpoint has an unsupported class mapping.")
        self.labels = [name.replace("Apple___", "").replace("_", " ")
                       for name, index in sorted(mapping.items(), key=lambda item: item[1])]
        self.classifier_transform = checkpoint_transform(metadata)
        self.classifier.requires_grad_(False).eval()
        self.classifier_name = next((name for name, path in CLASSIFIER_CHECKPOINTS.items()
                                     if path.resolve() == Path(classifier).resolve()), "Selected checkpoint")
        self.classifiers = {self.classifier_name: (self.classifier, self.classifier_transform)}
        for name, path in CLASSIFIER_CHECKPOINTS.items():
            if name == self.classifier_name or not path.is_file():
                continue
            model, saved = load_classifier(path, self.device)
            if (saved.get("architecture") != "efficientnet_b0" or saved.get("image_size") != 224
                    or saved.get("preprocessing") != "imagenet_rgb_resize" or saved.get("class_to_index") != mapping):
                raise ValueError(f"Incompatible comparison classifier: {path}")
            self.classifiers[name] = (model.requires_grad_(False).eval(), checkpoint_transform(saved))
        self.segmenters = {}
        for name, path in self.paths.items():
            model, checkpoint = load_segmenter(path, self.device)
            if checkpoint.get("architecture") != "unet_resnet34" or checkpoint.get("image_size") != 256:
                raise ValueError(f"Segmenter must be a ResNet34 U-Net 256px checkpoint: {path}")
            if "threshold" not in checkpoint:
                raise ValueError(f"Segmenter is missing its saved threshold: {path}")
            threshold = float(checkpoint["threshold"])
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError(f"Invalid saved segmentation threshold: {path}")
            self.segmenters[name] = (model.requires_grad_(False).eval(), threshold)

    @torch.inference_mode()
    def predict(self, image, segmenter=PRIMARY, classifier=None):
        image = prepare_image(image)
        if segmenter not in self.segmenters:
            raise ValueError("Choose one of the available segmentation models.")
        selected_model, transform = self.classifier, self.classifier_transform
        if classifier is not None:
            if classifier not in self.classifiers:
                raise ValueError("Choose one of the available classifiers.")
            selected_model, transform = self.classifiers[classifier]
        inputs = transform(image).unsqueeze(0).to(self.device)
        probabilities = selected_model(inputs).softmax(dim=1)[0].cpu().numpy()
        scores = dict(zip(self.labels, map(float, probabilities)))
        predicted = self.labels[int(probabilities.argmax())]
        confidence = float(probabilities.max())
        model, threshold = self.segmenters[segmenter]
        inputs = segmentation_input(image).unsqueeze(0).to(self.device)
        probability = model(inputs).sigmoid()[0, 0].cpu().numpy()
        overlay, mask = lesion_overlay(image, probability, threshold)
        area = float((probability >= threshold).mean())
        details = (
            f"**Classifier:** {classifier or getattr(self, 'classifier_name', 'Selected checkpoint')}  \n"
            f"**Predicted disease:** {predicted}  \n"
            f"**Softmax score:** {confidence:.2%} (uncalibrated; not diagnostic certainty)  \n"
            f"**Analyzed image dimensions:** {image.width} × {image.height} pixels  \n"
            f"**Segmentation model:** {segmenter}  \n"
            f"**Saved lesion threshold:** {threshold:.2f}  \n"
            f"**Highlighted area:** {area:.2%} of the 256 × 256 image grid. "
            "This is not a percentage of diseased leaf tissue."
        )
        if predicted == "healthy":
            if area > 0:
                details += (
                    "\n\n**Warning: model disagreement.** The classifier predicts healthy, "
                    "but the segmenter highlights pixels. These pixels are a separate "
                    "segmentation output, not a disease diagnosis; either model may be wrong. "
                    "The overlay remains visible for inspection."
                )
            else:
                details += (
                    "\n\nThe classifier predicts healthy and the segmenter highlights no pixels "
                    "at its saved threshold. This does not establish that the leaf is disease-free."
                )
        return scores, image, overlay, mask, details

    def predict_crop(self, editor, segmenter=PRIMARY, classifier=None):
        return self.predict(prepare_crop(editor), segmenter, classifier)

    def compare_crop(self, original, editor, segmenter=PRIMARY, classifier=None):
        full = self.predict(original, segmenter, classifier)
        cropped = self.predict_crop(editor, segmenter, classifier)
        rows = ["### Full image versus selected crop", "",
                "| Class | Full-image softmax score | Crop softmax score |",
                "| --- | ---: | ---: |"]
        for label in self.labels:
            rows.append(f"| {label} | {full[0][label]:.2%} | {cropped[0][label]:.2%} |")
        rows.extend(["", f"Full image: {full[1].width} × {full[1].height} pixels. "
                     f"Selected crop: {cropped[1].width} × {cropped[1].height} pixels.", "",
                     "The three panels show the selected crop. A changed or higher score "
                     "does not establish a more accurate prediction.", "",
                     "**Full-image result:**", full[4]])
        return (*cropped, "\n".join(rows))


def build_demo(pipeline):
    import gradio as gr

    with gr.Blocks(title="Apple leaf pipeline demo") as demo:
        gr.Markdown(
            "# Apple leaf disease and lesion demo\n"
            "Upload an image, crop around one main leaf, apply the crop, then analyze it. "
            "Keep the entire leaf and its visible lesions within the crop.\n\n"
            "**Research prototype — field-image performance is limited. This demo illustrates "
            "the pipeline; it does not provide reliable field diagnosis.** A softmax score "
            "can be high even for an incorrect prediction. Red regions are predicted lesions "
            "and may include healthy structures."
        )
        gr.Markdown(
            "## Did orchard fine-tuning improve recognition?\n"
            "Held-out evaluation · 475 PlantVillage images and 252 PP2020 orchard images · 22 Oct 2026\n\n"
            "| Classifier | PlantVillage accuracy | Orchard accuracy | Orchard macro-F1 |\n"
            "| --- | ---: | ---: | ---: |\n"
            "| Reference | 99.79% | 48.41% | 0.4347 |\n"
            "| Background-robust | 100.00% | 56.75% | 0.5528 |\n"
            "| **C1 field-adapted** | **99.79%** | **94.84%** | **0.9465** |\n\n"
            "**C1 improved orchard accuracy by 46.43 percentage points and matched reference PlantVillage accuracy.** "
            "C1 orchard recall: Scab **89.89%**, Cedar Rust **97.85%**, Healthy **97.14%**. "
            "Only **1 of 213** C1 orchard predictions with a score ≥ 90% was wrong, versus **25 of 87** for reference.\n\n"
            "Orchard macro-F1 covers three represented classes; all four prediction outputs were retained. "
            "PP2020 contains no black-rot examples, so field black-rot recall is unmeasured. "
            "These are dataset results, not a guarantee for a new upload. Select a classifier below to try the same image."
        )
        with gr.Row():
            with gr.Column():
                upload = gr.Image(type="pil", image_mode="RGB", format="png", sources=["upload"], label="Full uploaded image")
                editor = gr.ImageEditor(type="pil", image_mode="RGB", format="png", sources=[],
                                        transforms=["crop"], brush=False, eraser=False, layers=False,
                                        label="Crop around a single leaf, then apply the crop", height=400)
            with gr.Column():
                classifier_selection = gr.Dropdown(choices=list(pipeline.classifiers), value=pipeline.classifier_name,
                                                    label="Disease classifier", interactive=True)
                selection = gr.Radio(choices=list(pipeline.segmenters), value=PRIMARY, label="Lesion model")
                gr.Markdown(
                    "The reference favors diseased-image Dice. The 5% hard-negative comparison "
                    "produces fewer false-positive pixels on healthy images. "
                    "Both use their saved thresholds."
                )
                run = gr.Button("Analyze selected crop", variant="primary")
                compare = gr.Button("Compare crop with full image")
                full_run = gr.Button("Analyze full image")
                scores = gr.Label(num_top_classes=4, label="Softmax score")
                details = gr.Markdown()
        with gr.Row():
            original = gr.Image(type="pil", format="png", label="Analyzed leaf image (selected crop or full image)", interactive=False)
            mask = gr.Image(type="pil", image_mode="L", format="png", label="Predicted binary mask (white = selected pixels)", interactive=False)
            overlay = gr.Image(type="pil", format="png", label="Transparent red lesion overlay (separate segmentation output)", interactive=False)
        comparison = gr.Markdown()
        gr.Markdown(
            "EfficientNet-B0: 224 × 224 input · ResNet34 U-Net: 256 × 256 input. "
            "Both models receive the same selected crop using their own training preprocessing. "
            "The models run independently; healthy predictions do not suppress segmentation. "
            "The red overlay uses 45% opacity. All threshold-selected regions are retained."
        )

        def analyze(image, model, classifier=None):
            try:
                return pipeline.predict(image, model, classifier)
            except ValueError as error:
                raise gr.Error(str(error)) from error

        outputs = [scores, original, overlay, mask, details]
        def analyze_crop(value, model, classifier=None):
            try:
                return pipeline.predict_crop(value, model, classifier)
            except ValueError as error:
                raise gr.Error(str(error)) from error

        def compare_images(image, value, model, classifier=None):
            try:
                return pipeline.compare_crop(image, value, model, classifier)
            except ValueError as error:
                raise gr.Error(str(error)) from error

        run.click(analyze_crop, inputs=[editor, selection, classifier_selection], outputs=outputs,
                  concurrency_limit=1, concurrency_id="inference", api_name="analyze_crop")
        full_run.click(analyze, inputs=[upload, selection, classifier_selection], outputs=outputs,
                       concurrency_limit=1, concurrency_id="inference", api_name="analyze")
        compare.click(compare_images, inputs=[upload, editor, selection, classifier_selection], outputs=[*outputs, comparison],
                      concurrency_limit=1, concurrency_id="inference", api_name="compare_crop")
        # Clear previous results when either input changes, preventing stale output pairs.
        upload.change(seed_crop, inputs=upload, outputs=editor, queue=False, api_name="seed_crop")
        for component in [upload, editor, selection, classifier_selection]:
            component.change(lambda: [None, None, None, None, "", ""],
                             outputs=[*outputs, comparison], queue=False)
        # Switching analysis modes also clears a comparison made for a previous input.
        run.click(lambda: "", outputs=comparison, queue=False)
        full_run.click(lambda: "", outputs=comparison, queue=False)
        gr.ClearButton([upload, editor, scores, original, overlay, mask, details, comparison])
    return demo
