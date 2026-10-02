from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional


class LesionDataset(Dataset):
    """Paired RGB images and binary lesion masks listed in a CSV manifest."""

    def __init__(self, manifest, image_size=256, training=False):
        self.records = pd.read_csv(manifest)
        required = {"image_path", "mask_path"}
        if not required.issubset(self.records.columns):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")
        self.image_size = image_size
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        with Image.open(row["mask_path"]) as source:
            mask = source.convert("RGB")
        image = functional.resize(image, [self.image_size, self.image_size])
        mask = functional.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=functional.InterpolationMode.NEAREST,
        )
        if self.training and torch.rand(()) < 0.5:
            image, mask = functional.hflip(image), functional.hflip(mask)
        if self.training and torch.rand(()) < 0.5:
            image, mask = functional.vflip(image), functional.vflip(mask)
        image = functional.to_tensor(image)
        image = functional.normalize(
            image,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        mask_array = np.asarray(mask, dtype=np.uint8)
        chroma = mask_array.max(axis=2) - mask_array.min(axis=2)
        if chroma.max() >= 64:
            # Zenodo Apple masks encode the leaf as white, background as black,
            # and lesions as saturated red/green/blue. JPEG compression can
            # perturb exact RGB values, hence a chroma threshold is safer than
            # exact colour matching.
            lesion = chroma >= 64
        else:
            # Conventional binary masks: black background, white lesion.
            lesion = mask_array.max(axis=2) > 127
        mask = torch.from_numpy(lesion.astype(np.float32)).unsqueeze(0)
        return image, mask, str(row["image_path"])


def validate_manifest(manifest):
    manifest = Path(manifest)
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Lesion manifest not found: {manifest.resolve()}\n"
            "Training requires real pixel-level disease-lesion masks and CSV "
            "files named train.csv, val.csv, and test.csv. The existing "
            "PlantVillage segmented folder contains whole-leaf masks and must "
            "not be used as lesion ground truth. See LESION_DETECTION.md."
        )
    records = pd.read_csv(manifest)
    missing = []
    for column in ("image_path", "mask_path"):
        if column not in records:
            raise ValueError(f"Missing manifest column: {column}")
        missing.extend(path for path in records[column] if not Path(path).is_file())
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} files; examples: {missing[:5]}")
