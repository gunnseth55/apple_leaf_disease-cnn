from pathlib import Path

from PIL import Image


image_dir = Path("data/lesion_hard_negatives/images")
mask_dir = Path("data/lesion_hard_negatives/masks")
mask_dir.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
created = 0

for image_path in image_dir.iterdir():
    if image_path.suffix.lower() not in extensions:
        continue

    with Image.open(image_path) as image:
        mask = Image.new("L", image.size, color=0)

    mask_path = mask_dir / f"{image_path.stem}.png"
    mask.save(mask_path)
    print(f"Created: {mask_path}")
    created += 1

print(f"Created {created} empty masks")