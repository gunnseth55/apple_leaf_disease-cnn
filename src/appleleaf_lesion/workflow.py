from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import LesionDataset, validate_manifest
from .model import ResNet34UNet


def source_sampling_weights(records, real_fraction, hard_negative_fraction=0.0):
    """Assign source probability mass while preserving a real/synthetic ratio."""
    if not 0 < real_fraction < 1:
        raise ValueError("Real sampling fraction must be between 0 and 1")
    if not 0 <= hard_negative_fraction < 1:
        raise ValueError("Hard-negative sampling fraction must be in [0, 1)")
    if "source" not in records.columns:
        raise ValueError("Source-balanced sampling requires a 'source' column")

    sources = records["source"].astype(str).str.lower()
    supported = {"real", "synthetic"}
    if hard_negative_fraction:
        supported.add("hard_negative")
    unsupported = sorted(set(sources) - supported)
    if unsupported:
        raise ValueError(
            "Source-balanced sampling supports only real and synthetic rows; "
            f"found: {unsupported}"
        )
    real_count = int((sources == "real").sum())
    synthetic_count = int((sources == "synthetic").sum())
    hard_negative_count = int((sources == "hard_negative").sum())
    if not real_count or not synthetic_count:
        raise ValueError("Both real and synthetic training samples are required")
    if hard_negative_fraction and not hard_negative_count:
        raise ValueError("Hard-negative sampling requires hard_negative rows")

    lesion_fraction = 1 - hard_negative_fraction
    source_weights = {
        "real": lesion_fraction * real_fraction / real_count,
        "synthetic": lesion_fraction * (1 - real_fraction) / synthetic_count,
    }
    if hard_negative_fraction:
        source_weights["hard_negative"] = (
            hard_negative_fraction / hard_negative_count
        )

    return torch.tensor(
        [source_weights[source] for source in sources],
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
            train_dataset.records,
            config.real_sampling_fraction,
            config.hard_negative_sampling_fraction,
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=config.samples_per_epoch or len(train_dataset),
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


def build_healthy_loader(manifest, config):
    """Build a shuffled auxiliary loader containing only hard negatives."""
    validate_manifest(manifest)
    dataset = LesionDataset(manifest, config.image_size, training=True)
    if "source" not in dataset.records.columns:
        raise ValueError("Healthy auxiliary manifest requires a 'source' column")
    healthy_records = dataset.records[
        dataset.records["source"].astype(str).str.lower() == "hard_negative"
    ].reset_index(drop=True)
    if healthy_records.empty:
        raise ValueError("Healthy auxiliary manifest has no hard_negative rows")
    dataset.records = healthy_records
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed + 1),
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
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
