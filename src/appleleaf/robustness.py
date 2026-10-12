"""Training-only image perturbations with no external data or label changes."""

import hashlib
import io

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .datasets import build_transform, load_image_and_mask


ROBUST_AUGMENTATION = {
    "background": {"original_probability": 0.35,
                   "replacement": "equal probability: training-only nonleaf donor crop or synthetic neutral/earth/foliage texture",
                   "donor_max_leaf_fraction": 0.0, "donor_attempts": 8},
    "geometry": "flips; p=0.8 rotation +/-180 degrees with expanded canvas, fit entire image, scale 0.75-1.0, random placement; p=0.2 retain geometry",
    "colour": "p=0.8: brightness/contrast +/-0.30, saturation +/-0.15, hue +/-0.015",
    "gamma": "p=0.3: 0.75-1.30",
    "shadow": "p=0.3: smooth illumination gradient, darkest factor 0.70-0.90",
    "quality": "p=0.2 Gaussian blur radius 0.3-1.0; p=0.25 JPEG quality 60-95",
    "forbidden": "no crops, lesion erasing, synthetic damage, disease relabeling or Kashmir donors",
}


def uniform(low, high):
    return float(torch.empty(()).uniform_(low, high))


def smooth_texture(size, seed=None):
    """Nonuniform stress texture; not a claim of realistic orchard scenery."""
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    palette = np.asarray([(65, 95, 56), (112, 91, 67), (145, 145, 140),
                          (180, 170, 147), (170, 132, 149)], dtype=np.float32)
    indices = torch.randint(len(palette), (2,), generator=generator).numpy()
    colours = palette[indices]
    layers = []
    for side in (3, 12):
        pixels = torch.randint(0, 256, (side, side), generator=generator).numpy().astype(np.uint8)
        layers.append(np.asarray(Image.fromarray(pixels).resize(size, Image.Resampling.BICUBIC), dtype=np.float32))
    mix = (0.65 * layers[0] + 0.35 * layers[1])[..., None] / 255
    texture = colours[0] * mix + colours[1] * (1 - mix)
    return np.clip(texture * (0.8 + 0.4 * mix), 0, 255).astype(np.uint8)


def deterministic_texture(size, identity):
    seed = int.from_bytes(hashlib.sha256(str(identity).encode()).digest()[:8], "big") % (2**63)
    return smooth_texture(size, seed)


class RobustImageTransform:
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.colour = transforms.ColorJitter(brightness=.30, contrast=.30, saturation=.15, hue=.015)
        self.preprocess = build_transform(image_size, "imagenet_rgb_resize")

    def __call__(self, image):
        image = image.convert("RGB")
        if torch.rand(()) < .5:
            image = ImageOps.mirror(image)
        if torch.rand(()) < .5:
            image = ImageOps.flip(image)
        if torch.rand(()) < .8:
            image = image.rotate(uniform(-180, 180), resample=Image.Resampling.BICUBIC,
                                 expand=True, fillcolor=(127, 127, 127))
            side = max(1, round(self.image_size * uniform(.75, 1.0)))
            image = ImageOps.contain(image, (side, side), Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (self.image_size, self.image_size), (127, 127, 127))
            x = int(torch.randint(self.image_size - image.width + 1, ()))
            y = int(torch.randint(self.image_size - image.height + 1, ()))
            canvas.paste(image, (x, y))
            image = canvas
        else:
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        if torch.rand(()) < .8:
            image = self.colour(image)
        if torch.rand(()) < .3:
            image = TF.adjust_gamma(image, uniform(.75, 1.30))
        if torch.rand(()) < .3:
            values = np.asarray(image, dtype=np.float32)
            floor = uniform(.70, .90)
            axis = int(torch.randint(2, ()))
            gradient = np.linspace(floor, 1.0, self.image_size)
            if torch.rand(()) < .5:
                gradient = gradient[::-1]
            gradient = gradient[:, None, None] if axis == 0 else gradient[None, :, None]
            image = Image.fromarray(np.clip(values * gradient, 0, 255).astype(np.uint8))
        if torch.rand(()) < .2:
            image = image.filter(ImageFilter.GaussianBlur(uniform(.3, 1.0)))
        if torch.rand(()) < .25:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=int(torch.randint(60, 96, ())))
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                image = decoded.convert("RGB")
        return self.preprocess(image)


class RobustBackgroundDataset(Dataset):
    """All sources and all donor pixels come from the supplied PV training rows."""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        if "segmented_path" not in self.dataframe:
            raise ValueError("Robust background training requires PlantVillage leaf masks")
        self.transform = transform or RobustImageTransform()

    def __len__(self):
        return len(self.dataframe)

    def donor_background(self, size):
        # Draw donor independently of target label; reject patches containing
        # any foreground to avoid copying disease/leaf texture into backgrounds.
        for _ in range(8):
            donor = self.dataframe.iloc[int(torch.randint(len(self.dataframe), ()))]
            pixels, mask = load_image_and_mask(donor.path, donor.segmented_path)
            height, width = mask.shape
            patch_w, patch_h = max(4, width // 5), max(4, height // 5)
            patch_w, patch_h = min(patch_w, width), min(patch_h, height)
            x = int(torch.randint(width - patch_w + 1, ()))
            y = int(torch.randint(height - patch_h + 1, ()))
            if not mask[y:y + patch_h, x:x + patch_w].any():
                crop = Image.fromarray(pixels[y:y + patch_h, x:x + patch_w])
                return np.asarray(crop.resize(size, Image.Resampling.BICUBIC))
        return smooth_texture(size)

    def make_image(self, index):
        row = self.dataframe.iloc[index]
        pixels, mask = load_image_and_mask(row.path, row.segmented_path)
        if torch.rand(()) >= .35:
            size = (pixels.shape[1], pixels.shape[0])
            background = (self.donor_background(size) if torch.rand(()) < .5 else smooth_texture(size))
            pixels = np.where(mask[..., None], pixels, background).astype(np.uint8)
        return Image.fromarray(pixels)

    def __getitem__(self, index):
        return self.transform(self.make_image(index)), int(self.dataframe.iloc[index].label)
