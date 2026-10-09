from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


def grad_cam(model, image_tensor, device, target_class=None):
    """Return probabilities and a Grad-CAM map for one image.

    The map is computed from the final convolutional feature tensor. It is a
    coarse influence map, not a pixel-level lesion segmentation.
    """
    model.eval()
    inputs = image_tensor.unsqueeze(0).to(device)
    # Stop before the final pooling operation. This retains a 16x16 spatial
    # feature map instead of the coarser 8x8 tensor used by the classifier.
    features = model.features[:-1](inputs)
    features.retain_grad()
    final_features = model.features[-1](features)
    if hasattr(model, "avgpool"):
        final_features = model.avgpool(final_features).flatten(1)
    logits = model.classifier(final_features)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = int(logits.argmax(dim=1).item())
    explained_class = predicted_class if target_class is None else int(target_class)

    model.zero_grad(set_to_none=True)
    logits[0, explained_class].backward()
    channel_weights = features.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((channel_weights * features).sum(dim=1, keepdim=True))
    cam = functional.interpolate(
        cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    cam -= cam.min()
    maximum = cam.max()
    if maximum > 0:
        cam /= maximum

    return probabilities[0].detach().cpu().numpy(), cam.detach().cpu().numpy()


def attention_inside_mask(cam, mask):
    resized_mask = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (cam.shape[1], cam.shape[0]), Image.Resampling.NEAREST
        )
    ) > 0
    total = float(cam.sum())
    return float(cam[resized_mask].sum() / total) if total > 0 else float("nan")


def save_explanation(
    original,
    mask,
    cam,
    probabilities,
    class_names,
    true_class,
    predicted_class,
    source_path,
    output_path,
):
    display_cam = np.asarray(
        Image.fromarray(cam.astype(np.float32)).resize(
            (original.shape[1], original.shape[0]), Image.Resampling.BILINEAR
        )
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].imshow(original)
    axes[0].contour(mask, levels=[0.5], colors=["cyan"], linewidths=1)
    axes[0].set_title("Test image\nCyan: leaf-mask boundary")

    axes[1].imshow(original)
    heatmap = axes[1].imshow(display_cam, cmap="jet", alpha=0.48, vmin=0, vmax=1)
    axes[1].contour(mask, levels=[0.5], colors=["cyan"], linewidths=1)
    axes[1].set_title("Grad-CAM evidence\nRed/yellow: strongest influence")
    figure.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

    positions = np.arange(len(class_names))
    colours = ["#4c78a8"] * len(class_names)
    colours[predicted_class] = "#e45756"
    axes[2].barh(positions, probabilities, color=colours)
    axes[2].set_yticks(positions, class_names)
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("Predicted probability")
    axes[2].set_title("Class scores")
    axes[2].invert_yaxis()
    for position, probability in zip(positions, probabilities):
        axes[2].text(
            min(float(probability) + 0.015, 0.94),
            position,
            f"{probability:.1%}",
            va="center",
        )

    for axis in axes[:2]:
        axis.axis("off")
    status = "correct" if predicted_class == true_class else "incorrect"
    figure.suptitle(
        f"True: {class_names[true_class]} | Predicted: "
        f"{class_names[predicted_class]} ({probabilities[predicted_class]:.1%}) | {status}\n"
        f"Source: {Path(source_path).name}"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
