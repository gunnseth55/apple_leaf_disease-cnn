from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import LesionDataset, validate_manifest
from .model import ResNet34UNet


def source_sampling_weights(records, real_fraction):
    """Give real and synthetic samples the requested probability mass."""
    if not 0 < real_fraction < 1:
        raise ValueError("Real sampling fraction must be between 0 and 1")
    if "source" not in records.columns:
        raise ValueError("Source-balanced sampling requires a 'source' column")

    sources = records["source"].astype(str).str.lower()
    unsupported = sorted(set(sources) - {"real", "synthetic"})
    if unsupported:
        raise ValueError(
            "Source-balanced sampling supports only real and synthetic rows; "
            f"found: {unsupported}"
        )
    real_count = int((sources == "real").sum())
    synthetic_count = int((sources == "synthetic").sum())
    if not real_count or not synthetic_count:
        raise ValueError("Both real and synthetic training samples are required")

    return torch.tensor(
        [
            real_fraction / real_count
            if source == "real"
            else (1 - real_fraction) / synthetic_count
            for source in sources
        ],
        dtype=torch.double,
    )


def build_loaders(manifest_dir, config, train_manifest=None):
    manifest_dir = Path(manifest_dir)
    paths = {split: manifest_dir / f"{split}.csv" for split in ("train", "val", "test")}
    if train_manifest is not None:
        paths["train"] = Path(train_manifest)
    for path in paths.values():
        validate_manifest(path)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset = LesionDataset(paths["train"], config.image_size, training=True)
    if config.real_sampling_fraction is None:
        train_loader = DataLoader(
            train_dataset, shuffle=True, generator=generator, **common
        )
    else:
        weights = source_sampling_weights(
            train_dataset.records, config.real_sampling_fraction
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(train_dataset, sampler=sampler, **common)
    return (
        train_loader,
        DataLoader(
            LesionDataset(paths["val"], config.image_size), shuffle=False, **common
        ),
        DataLoader(
            LesionDataset(paths["test"], config.image_size), shuffle=False, **common
        ),
    )


def save_checkpoint(
    path,
    model,
    config,
    best_epoch,
    test_metrics,
    pretrained_encoder=True,
    threshold=None,
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
            "threshold": config.threshold if threshold is None else float(threshold),
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
