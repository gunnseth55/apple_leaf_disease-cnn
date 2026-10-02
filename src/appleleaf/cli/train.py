import argparse
from pathlib import Path

import numpy as np
import torch

from appleleaf.config import ExperimentConfig
from appleleaf.data import download_dataset
from appleleaf.engine import (
    choose_device,
    classification_metrics,
    predict,
    seed_everything,
    train_model,
)
from appleleaf.model import AppleLeafCNN
from appleleaf.plots import plot_confusion_matrix, plot_training_history
from appleleaf.workflow import build_loaders, class_names, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Apple leaf CNN")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument(
        "--randomize-backgrounds",
        action="store_true",
        help="randomize only training-image backgrounds using segmentation masks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir or (
        Path("artifacts/background_augmented")
        if args.randomize_backgrounds
        else Path("artifacts")
    )
    config = ExperimentConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        artifacts_dir=artifacts_dir,
    )
    seed_everything(config.seed)
    device = choose_device(args.device)
    dataset_path = args.dataset_path or download_dataset()
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")

    (train_loader, val_loader, test_loader), (train_df, _, _) = build_loaders(
        dataset_path,
        config,
        args.verify_images,
        randomize_train_backgrounds=args.randomize_backgrounds,
    )
    model = AppleLeafCNN().to(device)
    class_counts = train_df["label"].value_counts().reindex(range(4)).to_numpy()
    class_weights = torch.tensor(
        len(train_df) / (4 * class_counts), dtype=torch.float32, device=device
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history, best_epoch, best_val_f1 = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        config.epochs,
    )
    results = predict(model, test_loader, device)
    metrics = classification_metrics(results, class_names())

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(config.artifacts_dir / "training_history.csv", index=False)
    np.savez_compressed(config.artifacts_dir / "test_predictions.npz", **results)
    plot_training_history(history, config.artifacts_dir / "training_history.png")
    plot_confusion_matrix(
        results["labels"],
        results["predictions"],
        class_names(),
        config.artifacts_dir / "confusion_matrix.png",
    )
    checkpoint_name = (
        "apple_background_augmented_checkpoint.pt"
        if args.randomize_backgrounds
        else "apple_baseline_checkpoint.pt"
    )
    save_checkpoint(
        config.artifacts_dir / checkpoint_name,
        model,
        config,
        best_epoch,
        metrics,
        training_strategy=(
            "randomized_backgrounds" if args.randomize_backgrounds else "original"
        ),
    )
    print(f"Restored epoch {best_epoch}; validation macro F1: {best_val_f1:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}\n")
    print(metrics["report"])
    print(f"Artifacts saved to: {config.artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
