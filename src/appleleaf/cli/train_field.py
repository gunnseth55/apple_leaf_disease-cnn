"""Mixed-source EfficientNet fine-tuning. Training runs only without --dry-run."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score

from appleleaf.config import APPLE_CLASSES
from appleleaf.datasets import build_transform
from appleleaf.engine import choose_device, predict, run_epoch, seed_everything
from appleleaf.workflow import load_model
from appleleaf.cli.train_efficientnet import save_best, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sampling_weights(frame, field_fraction=0.5):
    weights = pd.Series(0., index=frame.index)
    for source, mass in [("plantvillage", 1-field_fraction), ("pp2020", field_fraction)]:
        selected = frame[frame.source == source]
        if selected.empty:
            raise ValueError(f"Missing training source: {source}")
        classes = selected.label.unique()
        for label in classes:
            indices = selected[selected.label == label].index
            weights.loc[indices] = mass / len(classes) / len(indices)
    return torch.tensor(weights.to_numpy(), dtype=torch.double)


def source_metrics(results, present_labels):
    return {"accuracy": float(accuracy_score(results["labels"], results["predictions"])),
            "macro_f1": float(f1_score(results["labels"], results["predictions"], labels=present_labels, average="macro", zero_division=0)),
            "macro_f1_all_four": float(f1_score(results["labels"], results["predictions"], labels=[0,1,2,3], average="macro", zero_division=0))}


class FieldDataset(Dataset):
    def __init__(self, frame, training=False):
        self.frame = frame.reset_index(drop=True)
        transform = build_transform(224, "imagenet_rgb_resize")
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.15, hue=.01), transform,
        ]) if training else transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image), int(row.label)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/efficientnet_b0/apple_efficientnet_b0_checkpoint.pt"))
    parser.add_argument("--field-split-dir", type=Path, default=Path("data/field_manifests/pp2020_split_v1"))
    parser.add_argument("--pv-manifest-dir", type=Path, default=Path("artifacts/efficientnet_b0"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/efficientnet_b0_field_c1_v1"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--field-fraction", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or not 0 < args.field_fraction < 1:
        parser.error("Positive epochs/batch size, nonnegative workers and field fraction between 0 and 1 required")
    if any(not math.isfinite(value) or value <= 0 for value in [args.head_lr, args.backbone_lr]):
        parser.error("Learning rates must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Weight decay must be finite and nonnegative")
    if args.artifacts_dir.exists():
        parser.error("Use a new artifacts directory; existing runs and reference weights are preserved")
    for protected in [args.checkpoint.parent, args.field_split_dir, args.pv_manifest_dir]:
        if args.artifacts_dir.resolve() == protected.resolve() or protected.resolve() in args.artifacts_dir.resolve().parents:
            parser.error("Training artifacts must be separate from reference/split inputs")
    return args


def prepare_data(args):
    project = Path(__file__).resolve().parents[3]
    subprocess.run([sys.executable, str(project / "scripts/build_pp2020_field_splits.py"),
                    "--project", str(project), "--output-dir", str(args.field_split_dir.resolve()), "--verify"], check=True)
    paths = {"pv_train": args.pv_manifest_dir / "train_manifest.csv", "pv_val": args.pv_manifest_dir / "validation_manifest.csv",
             "pv_test_reserved": args.pv_manifest_dir / "test_manifest.csv", "field_train": args.field_split_dir / "train.csv",
             "field_val": args.field_split_dir / "val.csv", "field_test_reserved": args.field_split_dir / "test.csv"}
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    mapping = dict(zip(APPLE_CLASSES, range(4)))
    seen_paths, seen_pixels = {}, {}
    for name, frame in frames.items():
        if frame.path.duplicated().any():
            raise ValueError(f"Duplicate paths: {name}")
        if not frame.class_name.map(mapping).eq(frame.label).all():
            raise ValueError(f"Class mapping mismatch: {name}")
        for row in frame.itertuples():
            key = str(Path(row.path).resolve())
            if not Path(key).is_file():
                raise FileNotFoundError(key)
            # Recorded original PV pixel hashes and PP2020 normalized pixel hashes
            # screen all train/validation/test manifests without test inference.
            pixel = getattr(row, "pixel_sha256", getattr(row, "image_hash", None))
            if not pixel:
                raise ValueError(f"Missing audited pixel identity: {key}")
            if key in seen_paths or pixel in seen_pixels:
                raise ValueError(f"Repeated path/pixel content across manifests: {name}: {key}")
            seen_paths[key], seen_pixels[pixel] = name, name
        if not name.endswith("reserved"):
            frame["source"] = "pp2020" if name.startswith("field") else "plantvillage"
    if set(frames["pv_train"].label) != {0,1,2,3} or set(frames["field_train"].label) != {0,2,3}:
        raise ValueError("Expected four PV classes and three PP2020 classes")
    mixed = pd.concat([frames["pv_train"], frames["field_train"]], ignore_index=True)
    inputs = {str(path.resolve()): digest(path) for path in paths.values()}
    for path in [args.field_split_dir / "split_report.json", args.field_split_dir / "split_integrity_report.json"]:
        inputs[str(path.resolve())] = digest(path)
    return mixed, {"plantvillage": frames["pv_val"], "pp2020": frames["field_val"]}, inputs


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    reference_hash = digest(args.checkpoint)
    mixed, validation, inputs = prepare_data(args)
    model, metadata = load_model(args.checkpoint, device)
    if (metadata.get("architecture") != "efficientnet_b0" or metadata.get("preprocessing") != "imagenet_rgb_resize"
        or metadata.get("image_size") != 224 or metadata.get("class_to_index") != dict(zip(APPLE_CLASSES, range(4)))):
        raise ValueError("Initialization must be the four-class 224px ImageNet-normalized EfficientNet-B0 checkpoint")
    weights = sampling_weights(mixed, args.field_fraction)
    plan = dict(architecture="efficientnet_b0", image_size=224, preprocessing="imagenet_rgb_resize",
                class_to_index=dict(zip(APPLE_CLASSES, range(4))), seed=args.seed,
                training_strategy="plantvillage_pp2020_source_and_within_source_class_balanced_finetuning",
                initialization_checkpoint=str(args.checkpoint.resolve()), initialization_sha256=reference_hash,
                source_manifest_sha256=inputs,
                source_training_counts=mixed.source.value_counts().to_dict(),
                validation_counts={name: len(frame) for name, frame in validation.items()},
                field_sampling_fraction=args.field_fraction, draws_per_epoch=len(mixed),
                sampling="replacement; source mass 50/50 by default; equal class mass within each source; unweighted cross entropy",
                selection="maximum mean of PV four-class macro-F1 and PP2020 present-three-class macro-F1; earliest tie",
                test_policy="No test loader or test inference; separate explicit evaluation after training",
                training_augmentation="full-image horizontal/vertical flips and mild color jitter; bicubic resize; ImageNet normalization; no random crops",
                config={"epochs":args.epochs, "batch_size":args.batch_size, "backbone_lr":args.backbone_lr,
                        "head_lr":args.head_lr, "weight_decay":args.weight_decay},
                limitations="Black rot supervision comes only from PlantVillage; no claim of orchard black-rot learning")
    if args.dry_run:
        if digest(args.checkpoint) != reference_hash:
            raise RuntimeError("Reference checkpoint changed")
        print(json.dumps(plan, indent=2))
        print("DRY RUN PASS: no epochs, optimizer steps, inference, checkpoint writes or output directory creation.")
        return
    args.artifacts_dir.mkdir(parents=True, exist_ok=False)
    output = args.artifacts_dir
    write_json(output / "run_config.json", plan)
    mixed.to_csv(output / "train_manifest.csv", index=False)
    for name, frame in validation.items():
        frame.to_csv(output / f"{name}_validation_manifest.csv", index=False)
    train_loader = DataLoader(FieldDataset(mixed, training=True), batch_size=args.batch_size, num_workers=args.num_workers,
                              sampler=WeightedRandomSampler(weights, len(mixed), replacement=True,
                              generator=torch.Generator().manual_seed(args.seed)))
    val_loaders = {name: DataLoader(FieldDataset(frame), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                   for name, frame in validation.items()}
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([{"params":model.features.parameters(), "lr":args.backbone_lr},
                                  {"params":model.classifier.parameters(), "lr":args.head_lr}], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    history, best_score, best_epoch = [], -1., None
    checkpoint = output / "apple_efficientnet_b0_field_c1_checkpoint.pt"
    for epoch in range(1, args.epochs+1):
        if any(digest(path) != expected for path, expected in inputs.items()) or digest(args.checkpoint) != reference_hash:
            raise RuntimeError("A frozen input changed during training")
        started = time.perf_counter()
        train = run_epoch(model, train_loader, criterion, device, optimizer)
        metrics = {name: source_metrics(predict(model, loader, device), [0,1,2,3] if name == "plantvillage" else [0,2,3])
                   for name, loader in val_loaders.items()}
        score = float(np.mean([result["macro_f1"] for result in metrics.values()]))
        row = {"epoch": epoch, "seconds":time.perf_counter()-started, "selection_score":score,
               **{f"train_{name}":float(value) for name,value in train.items()},
               **{f"{source}_val_{name}":value for source,result in metrics.items() for name,value in result.items()}}
        history.append(row)
        if score > best_score:
            best_score, best_epoch = score, epoch
            save_best(checkpoint, model, {**plan, "best_epoch":epoch, "best_validation_score":score, "source_validation":metrics})
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(f"Epoch {epoch:02d}/{args.epochs} | PV val F1 {metrics['plantvillage']['macro_f1']:.4f} | "
              f"Field val F1 {metrics['pp2020']['macro_f1']:.4f} | Selection {score:.4f} | {row['seconds']:.1f}s", flush=True)
        scheduler.step()
    if digest(args.checkpoint) != reference_hash or any(digest(path) != expected for path,expected in inputs.items()):
        raise RuntimeError("A frozen input changed")
    write_json(output / "training_summary.json", {"training_complete":True, "best_epoch":best_epoch,
               "best_validation_score":best_score, "checkpoint":str(checkpoint.resolve()),
               "reference_unchanged":True, "frozen_manifests_unchanged":True, "test_evaluated":False})
    print(f"Training complete. Best epoch {best_epoch}. Saved: {checkpoint}. Test sets remain unevaluated.")


if __name__ == "__main__":
    main()
