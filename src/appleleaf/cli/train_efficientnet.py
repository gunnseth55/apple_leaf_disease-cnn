"""PlantVillage-only EfficientNet-B0 transfer learning; run explicitly to train."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms

from appleleaf.config import APPLE_CLASSES, ExperimentConfig
from appleleaf.datasets import build_transform
from appleleaf.engine import choose_device, classification_metrics, predict, run_epoch, seed_everything
from appleleaf.model import build_model
from appleleaf.plots import plot_confusion_matrix, plot_training_history
from appleleaf.workflow import build_loaders, class_names


PREPROCESSING = "imagenet_rgb_resize"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True,
                        help="PlantVillage root containing the Apple colour folders")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/efficientnet_b0"))
    parser.add_argument("--epochs", type=int, default=15, help="Total epochs including head warmup")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify splits and print the plan; no model download or training")
    args = parser.parse_args(argv)
    if args.epochs < 1 or not 0 <= args.warmup_epochs < args.epochs:
        parser.error("epochs must be positive and warmup epochs must be in [0, epochs)")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and workers nonnegative")
    rates = (args.head_lr, args.backbone_lr, args.finetune_head_lr)
    if any(not np.isfinite(rate) or rate <= 0 for rate in rates):
        parser.error("learning rates must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight decay must be finite and nonnegative")
    if args.artifacts_dir.exists() and (
        not args.artifacts_dir.is_dir() or any(args.artifacts_dir.iterdir())
    ):
        parser.error("artifacts directory must be empty; use a new folder for another run")
    return args


def set_training_phase(model, warmup):
    for parameter in model.features.parameters():
        parameter.requires_grad_(not warmup)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)


def write_json(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")


def save_best(path, model, metadata):
    checkpoint = {
        **metadata,
        "model_state_dict": {key: value.detach().cpu().clone()
                             for key, value in model.state_dict().items()},
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    config = ExperimentConfig(image_size=224, batch_size=args.batch_size,
                              epochs=args.epochs, learning_rate=args.backbone_lr,
                              weight_decay=args.weight_decay, seed=args.seed,
                              num_workers=args.num_workers, artifacts_dir=args.artifacts_dir)
    # Preserve the entire image so peripheral lesions are not removed by crops.
    # Validation/test preprocessing is deterministic and contains no augmentation.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        build_transform(224, PREPROCESSING),
    ])
    loaders, frames = build_loaders(args.dataset_path, config, verify=True,
                                    preprocessing=PREPROCESSING,
                                    train_transform=train_transform)
    train_loader, val_loader, test_loader = loaders
    train_df, val_df, test_df = frames
    plan = {
        "architecture": "efficientnet_b0", "pretrained_weights": "IMAGENET1K_V1",
        "preprocessing": PREPROCESSING, "image_size": 224,
        "class_to_index": {name: i for i, name in enumerate(APPLE_CLASSES)},
        "seed": args.seed, "config": config.to_dict(),
        "dataset_path": str(args.dataset_path.resolve()),
        "split_counts": dict(zip(("train", "validation", "test"), map(len, frames))),
        "warmup_epochs": args.warmup_epochs,
        "head_lr": args.head_lr, "backbone_lr": args.backbone_lr,
        "finetune_head_lr": args.finetune_head_lr,
        "optimizer": "AdamW", "finetune_scheduler": "CosineAnnealingLR",
        "selection": "highest validation macro-F1; earliest epoch wins ties",
        "training_strategy": "plantvillage_only_imagenet_transfer",
        "training_augmentation": "horizontal/vertical flips, rotation +/-15 degrees, mild colour jitter",
        "preprocessing_details": "RGB; full-image bicubic resize 224x224; ToTensor; ImageNet mean/std; no crop",
        "loss": "CrossEntropyLoss with inverse-frequency weights from training split only",
        "device": str(device), "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete: no model weights downloaded and no epochs run.", flush=True)
        return

    output = args.artifacts_dir
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(("train", "validation", "test"), frames):
        frame.to_csv(output / f"{name}_manifest.csv", index=False)
    write_json(output / "run_config.json", plan)
    model = build_model("efficientnet_b0", pretrained=True).to(device)
    counts = train_df.label.value_counts().reindex(range(4)).to_numpy()
    weights = torch.tensor(len(train_df) / (4 * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    best_f1, best_epoch, history = -1.0, None, []
    checkpoint_path = output / "apple_efficientnet_b0_checkpoint.pt"
    optimizer, scheduler = None, None
    for epoch in range(1, args.epochs + 1):
        warmup = epoch <= args.warmup_epochs
        if epoch == 1 or epoch == args.warmup_epochs + 1:
            # Start fine-tuning from the best warmup checkpoint, if there was one.
            if not warmup and best_epoch is not None:
                saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(saved["model_state_dict"])
                del saved
            set_training_phase(model, warmup)
            groups = ([{"params": model.classifier.parameters(), "lr": args.head_lr}]
                      if warmup else [
                          {"params": model.features.parameters(), "lr": args.backbone_lr},
                          {"params": model.classifier.parameters(), "lr": args.finetune_head_lr},
                      ])
            optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
            scheduler = (None if warmup else torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.warmup_epochs))
        started = time.perf_counter()
        rates = [group["lr"] for group in optimizer.param_groups]
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer,
                                  freeze_features=warmup)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "phase": "head" if warmup else "finetune",
                        "backbone_lr": 0.0 if warmup else rates[0], "head_lr": rates[-1],
                        "seconds": time.perf_counter() - started,
                        **{f"train_{key}": value for key, value in train_metrics.items()},
                        **{f"val_{key}": value for key, value in val_metrics.items()}})
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            save_best(checkpoint_path, model, {**plan, "best_epoch": epoch,
                                               "best_val_macro_f1": best_f1})
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(f"Epoch {epoch:02d}/{args.epochs} ({history[-1]['phase']}) | "
              f"Train loss {train_metrics['loss']:.4f} | Val loss {val_metrics['loss']:.4f} | "
              f"Val acc {val_metrics['accuracy']:.4f} | Val F1 {val_metrics['macro_f1']:.4f} | "
              f"{history[-1]['seconds']:.1f}s", flush=True)
        if scheduler is not None:
            scheduler.step()

    # Test is touched for inference only after selection is complete.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    results = predict(model, test_loader, device)
    metrics = classification_metrics(results, class_names())
    checkpoint["test_accuracy"] = float(metrics["accuracy"])
    checkpoint["test_macro_f1"] = float(metrics["macro_f1"])
    checkpoint["training_complete"] = True
    save_best(checkpoint_path, model, {key: value for key, value in checkpoint.items()
                                     if key != "model_state_dict"})
    write_json(output / "metrics.json", {**metrics, "best_epoch": best_epoch,
                                          "best_val_macro_f1": best_f1})
    np.savez_compressed(output / "test_predictions.npz", **results)
    predictions = test_df.copy()
    predictions["prediction"] = results["predictions"]
    for i, name in enumerate(APPLE_CLASSES):
        predictions[f"probability_{name}"] = results["probabilities"][:, i]
    predictions.to_csv(output / "test_predictions.csv", index=False)
    plot_training_history(pd.DataFrame(history), output / "training_history.png")
    plot_confusion_matrix(results["labels"], results["predictions"], class_names(),
                           output / "confusion_matrix.png",
                           title="EfficientNet-B0 - Test Confusion Matrix")
    print(f"Restored epoch {best_epoch}; validation macro-F1 {best_f1:.4f}")
    print(f"Test accuracy {metrics['accuracy']:.4f}; macro-F1 {metrics['macro_f1']:.4f}")
    print(metrics["report"])
    print(f"Artifacts saved to {output.resolve()}")


if __name__ == "__main__":
    main()
