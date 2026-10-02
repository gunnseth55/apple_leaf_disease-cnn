from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import LesionDataset, validate_manifest
from .model import ResNet34UNet


def build_loaders(manifest_dir, config):
    manifest_dir = Path(manifest_dir)
    paths = {split: manifest_dir / f"{split}.csv" for split in ("train", "val", "test")}
    for path in paths.values():
        validate_manifest(path)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    generator = torch.Generator().manual_seed(config.seed)
    return (
        DataLoader(
            LesionDataset(paths["train"], config.image_size, training=True),
            shuffle=True,
            generator=generator,
            **common,
        ),
        DataLoader(
            LesionDataset(paths["val"], config.image_size), shuffle=False, **common
        ),
        DataLoader(
            LesionDataset(paths["test"], config.image_size), shuffle=False, **common
        ),
    )


def save_checkpoint(
    path, model, config, best_epoch, test_metrics, pretrained_encoder=True
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "architecture": "unet_resnet34",
            "pretrained_encoder": pretrained_encoder,
            "image_size": config.image_size,
            "threshold": config.threshold,
            "best_epoch": best_epoch,
            "test_metrics": test_metrics,
            "config": config.to_dict(),
        },
        path,
    )


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ResNet34UNet(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint
