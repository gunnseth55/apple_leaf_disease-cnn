import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from appleleaf.config import ExperimentConfig
from appleleaf.data import attach_segmented_paths, download_dataset, prepare_splits
from appleleaf.datasets import build_transform, load_image_and_mask
from appleleaf.engine import choose_device
from appleleaf.explainability import attention_inside_mask, grad_cam, save_explanation
from appleleaf.workflow import class_names, load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create per-image Grad-CAM explanations on the held-out test split"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/explanations"))
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2,
        help="when --indices is omitted, select this many test images per class",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    config = ExperimentConfig(
        image_size=int(checkpoint["image_size"]), seed=int(checkpoint["seed"])
    )
    dataset_path = args.dataset_path or download_dataset()
    _, _, test_df = prepare_splits(dataset_path, config.seed)
    test_df = attach_segmented_paths(test_df, dataset_path)
    transform = build_transform(config.image_size)
    names = class_names()
    rows = []

    indices = args.indices
    if indices is None:
        indices = []
        for label in range(len(names)):
            indices.extend(
                test_df.index[test_df["label"] == label][
                    : args.samples_per_class
                ].tolist()
            )

    invalid = [index for index in indices if index < 0 or index >= len(test_df)]
    if invalid:
        raise ValueError(f"Test indices out of range 0..{len(test_df) - 1}: {invalid}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        row = test_df.iloc[index]
        original, mask = load_image_and_mask(row["path"], row["segmented_path"])
        image = Image.fromarray(original)
        probabilities, cam = grad_cam(model, transform(image), device)
        predicted_class = int(probabilities.argmax())
        true_class = int(row["label"])
        leaf_attention = attention_inside_mask(cam, mask)
        output_path = args.output_dir / f"test_{index:04d}_{Path(row['path']).stem}.png"
        save_explanation(
            original,
            mask,
            cam,
            probabilities,
            names,
            true_class,
            predicted_class,
            row["path"],
            output_path,
        )
        result = {
            "test_index": index,
            "source_path": row["path"],
            "true_class": names[true_class],
            "predicted_class": names[predicted_class],
            "correct": predicted_class == true_class,
            "confidence": float(probabilities[predicted_class]),
            "attention_inside_leaf": leaf_attention,
            "attention_outside_leaf": 1.0 - leaf_attention,
            "explanation_path": str(output_path.resolve()),
        }
        result.update(
            {f"probability_{name.replace(' ', '_')}": float(probabilities[i]) for i, name in enumerate(names)}
        )
        rows.append(result)
        print(
            f"[{index}] true={names[true_class]!r} predicted={names[predicted_class]!r} "
            f"confidence={probabilities[predicted_class]:.3f} "
            f"attention_inside_leaf={leaf_attention:.3f}"
        )

    summary_path = args.output_dir / "explanations.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Explanations saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
