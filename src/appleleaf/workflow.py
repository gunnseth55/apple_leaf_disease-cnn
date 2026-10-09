from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import APPLE_CLASSES, ExperimentConfig
from .data import attach_segmented_paths, prepare_splits
from .datasets import AppleLeafDataset, RandomBackgroundDataset, build_transform
from .model import build_model


def class_names() -> list[str]:
    return [name.replace("Apple___", "").replace("_", " ") for name in APPLE_CLASSES]


def build_loaders(
    dataset_path,
    config: ExperimentConfig,
    verify=False,
    randomize_train_backgrounds=False,
    preprocessing="rgb_resize_totensor",
    train_transform=None,
):
    train_df, val_df, test_df = prepare_splits(dataset_path, config.seed, verify)
    if randomize_train_backgrounds:
        train_df = attach_segmented_paths(train_df, dataset_path)
    transform = build_transform(config.image_size, preprocessing)
    train_transform = train_transform if train_transform is not None else transform
    generator = torch.Generator().manual_seed(config.seed)
    common = {"batch_size": config.batch_size, "num_workers": config.num_workers}
    train_dataset = (
        RandomBackgroundDataset(train_df, train_transform)
        if randomize_train_backgrounds
        else AppleLeafDataset(train_df, train_transform)
    )
    loaders = (
        DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            **common,
        ),
        DataLoader(AppleLeafDataset(val_df, transform), shuffle=False, **common),
        DataLoader(AppleLeafDataset(test_df, transform), shuffle=False, **common),
    )
    return loaders, (train_df, val_df, test_df)


def save_checkpoint(path, model, config, best_epoch, metrics, training_strategy="original"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
            "class_to_index": {name: index for index, name in enumerate(APPLE_CLASSES)},
            "image_size": config.image_size,
            "seed": config.seed,
            "best_epoch": best_epoch,
            "test_accuracy": float(metrics["accuracy"]),
            "test_macro_f1": float(metrics["macro_f1"]),
            "training_strategy": training_strategy,
            "config": config.to_dict(),
        },
        path,
    )


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint.get("architecture", "appleleaf_cnn"),
                        num_classes=len(checkpoint["class_to_index"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
