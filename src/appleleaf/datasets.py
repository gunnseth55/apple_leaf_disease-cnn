from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


BACKGROUND_COLOURS = {
    "original": None,
    "white": (255, 255, 255),
    "blue": (60, 120, 200),
    "black": (0, 0, 0),
}

AUDIT_CONDITIONS = (*BACKGROUND_COLOURS, "background_only")


def build_transform(image_size: int, preprocessing="rgb_resize_totensor"):
    if preprocessing == "imagenet_rgb_resize":
        return transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    if preprocessing != "rgb_resize_totensor":
        raise ValueError(f"Unknown preprocessing: {preprocessing}")
    return transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )


def checkpoint_transform(checkpoint):
    return build_transform(int(checkpoint["image_size"]),
                           checkpoint.get("preprocessing", "rgb_resize_totensor"))


class AppleLeafDataset(Dataset):
    def __init__(self, dataframe, transform: Callable | None = None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


class RandomBackgroundDataset(Dataset):
    """Training dataset that makes the background unreliable as a class cue."""

    def __init__(self, dataframe, transform: Callable, original_probability: float = 0.25):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.original_probability = original_probability

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        original, mask = load_image_and_mask(row["path"], row["segmented_path"])

        # Retain some unmodified examples so normal photographs remain in the
        # training distribution. The rest receive a new background each visit.
        if np.random.random() < self.original_probability:
            changed = original
        elif (background_kind := np.random.randint(3)) == 0:
            palette = np.asarray(
                [(0, 0, 0), (255, 255, 255), (60, 120, 200)], dtype=np.uint8
            )
            changed = replace_background(
                original, mask, palette[np.random.randint(len(palette))]
            )
        elif background_kind == 1:
            colour = np.random.randint(0, 256, size=3, dtype=np.uint8)
            changed = replace_background(original, mask, colour)
        else:
            texture_small = Image.fromarray(
                np.random.randint(0, 256, size=(8, 8, 3), dtype=np.uint8)
            )
            texture = np.asarray(
                texture_small.resize(
                    (original.shape[1], original.shape[0]), Image.Resampling.BILINEAR
                )
            )
            changed = np.where(mask[..., None], original, texture).astype(np.uint8)

        return self.transform(Image.fromarray(changed)), int(row["label"])


def load_image_and_mask(original_path, segmented_path):
    with Image.open(original_path) as image:
        original = np.array(image.convert("RGB"))
    with Image.open(segmented_path) as image:
        segmented = np.array(image.convert("RGB"))
    if original.shape != segmented.shape:
        raise RuntimeError("Original and segmented images have different dimensions")
    mask = segmented.max(axis=2) > 10
    return original, mask


def replace_background(original, mask, colour):
    background = np.empty_like(original)
    background[:] = colour
    return np.where(mask[..., None], original, background)


class BackgroundTestDataset(Dataset):
    def __init__(self, dataframe, background_name: str, transform: Callable):
        if background_name not in AUDIT_CONDITIONS:
            raise ValueError(f"Unknown background {background_name!r}")
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.background_name = background_name
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        with Image.open(row["path"]) as image:
            original_image = image.convert("RGB")
        if self.background_name == "original":
            final_image = original_image
        else:
            original, mask = load_image_and_mask(row["path"], row["segmented_path"])
            if self.background_name == "background_only":
                # Retain the real scene outside the segmentation and remove all
                # leaf texture/colour. A mid-grey fill avoids introducing an
                # extreme black or white region while preserving the mask edge.
                changed = np.where(mask[..., None], 127, original).astype(np.uint8)
            else:
                changed = replace_background(
                    original, mask, BACKGROUND_COLOURS[self.background_name]
                )
            final_image = Image.fromarray(changed)
        return self.transform(final_image), int(row["label"])
