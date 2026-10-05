import argparse
from pathlib import Path

import torch

from appleleaf.engine import choose_device, seed_everything
from appleleaf_lesion.config import LesionConfig
from appleleaf_lesion.engine import calibrate_threshold, run_epoch, train_model
from appleleaf_lesion.losses import DiceFocalLoss
from appleleaf_lesion.model import ResNet34UNet
from appleleaf_lesion.workflow import build_loaders, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net ResNet34 lesion segmentation")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=None,
        help="Optional training CSV override; validation and test still use manifest-dir",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/lesion_detection"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--real-sampling-fraction",
        type=float,
        default=None,
        help="Sample this fraction of each training epoch from source=real rows",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.real_sampling_fraction is not None and not (
        0 < args.real_sampling_fraction < 1
    ):
        raise SystemExit("--real-sampling-fraction must be between 0 and 1")
    config = LesionConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        real_sampling_fraction=args.real_sampling_fraction,
        artifacts_dir=args.artifacts_dir,
    )
    seed_everything(config.seed)
    device = choose_device(args.device)
    try:
        train_loader, val_loader, test_loader = build_loaders(
            args.manifest_dir, config, train_manifest=args.train_manifest
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if config.real_sampling_fraction is not None:
        print(
            f"Source-balanced sampling: {config.real_sampling_fraction:.0%} real / "
            f"{1 - config.real_sampling_fraction:.0%} synthetic"
        )
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
    threshold, calibrated_val_metrics = calibrate_threshold(model, val_loader, device)
    test_metrics = run_epoch(model, test_loader, criterion, device, threshold)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(config.artifacts_dir / "training_history.csv", index=False)
    save_checkpoint(
        config.artifacts_dir / "unet_resnet34_lesion_checkpoint.pt",
        model,
        config,
        best_epoch,
        test_metrics,
        pretrained_encoder=not args.no_pretrained,
        threshold=threshold,
    )
    print(f"Restored epoch {best_epoch}; validation Dice: {best_dice:.4f}")
    print(
        f"Calibrated threshold: {threshold:.2f}; "
        f"validation Dice: {calibrated_val_metrics['dice']:.4f}"
    )
    print("Test: " + " | ".join(f"{key}={value:.4f}" for key, value in test_metrics.items()))


if __name__ == "__main__":
    main()
