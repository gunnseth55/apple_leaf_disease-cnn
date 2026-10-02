import argparse
from pathlib import Path

import torch

from appleleaf.engine import choose_device, seed_everything
from appleleaf_lesion.config import LesionConfig
from appleleaf_lesion.engine import run_epoch, train_model
from appleleaf_lesion.losses import DiceFocalLoss
from appleleaf_lesion.model import ResNet34UNet
from appleleaf_lesion.workflow import build_loaders, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net ResNet34 lesion segmentation")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/lesion_detection"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = LesionConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        artifacts_dir=args.artifacts_dir,
    )
    seed_everything(config.seed)
    device = choose_device(args.device)
    try:
        train_loader, val_loader, test_loader = build_loaders(args.manifest_dir, config)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    model = ResNet34UNet(pretrained=not args.no_pretrained).to(device)
    criterion = DiceFocalLoss(
        config.dice_weight,
        config.focal_weight,
        config.focal_alpha,
        config.focal_gamma,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history, best_epoch, best_dice = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, config
    )
    test_metrics = run_epoch(model, test_loader, criterion, device, config.threshold)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(config.artifacts_dir / "training_history.csv", index=False)
    save_checkpoint(
        config.artifacts_dir / "unet_resnet34_lesion_checkpoint.pt",
        model,
        config,
        best_epoch,
        test_metrics,
        pretrained_encoder=not args.no_pretrained,
    )
    print(f"Restored epoch {best_epoch}; validation Dice: {best_dice:.4f}")
    print("Test: " + " | ".join(f"{key}={value:.4f}" for key, value in test_metrics.items()))


if __name__ == "__main__":
    main()
