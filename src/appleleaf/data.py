import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from .config import APPLE_CLASSES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DATASET_HANDLE = "abdallahalidev/plantvillage-dataset"


def download_dataset() -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(DATASET_HANDLE))


def find_variant_root(dataset_path: str | Path, variant: str) -> Path:
    dataset_path = Path(dataset_path)
    candidates = [
        folder
        for folder in dataset_path.rglob("*")
        if folder.is_dir()
        and folder.name.lower() == variant.lower()
        and all((folder / class_name).is_dir() for class_name in APPLE_CLASSES)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one {variant!r} folder containing all Apple classes; "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def build_image_table(color_root: str | Path) -> pd.DataFrame:
    color_root = Path(color_root)
    class_to_index = {name: index for index, name in enumerate(APPLE_CLASSES)}
    records = []
    for class_name in APPLE_CLASSES:
        paths = sorted(
            path
            for path in (color_root / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise RuntimeError(f"No images found for {class_name}")
        records.extend(
            {
                "path": str(path),
                "filename": path.name,
                "class_name": class_name,
                "label": class_to_index[class_name],
            }
            for path in paths
        )
    return pd.DataFrame(records)


def verify_images(dataframe: pd.DataFrame) -> None:
    bad_images = []
    for image_path in dataframe["path"]:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as error:
            bad_images.append((image_path, str(error)))
    if bad_images:
        raise RuntimeError(f"Unreadable images found. First examples: {bad_images[:5]}")


def image_hash(image_path: str | Path) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        payload = str(image.size).encode() + image.tobytes()
    return hashlib.sha256(payload).hexdigest()


def remove_exact_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    result["image_hash"] = result["path"].apply(image_hash)
    label_counts = result.groupby("image_hash")["label"].nunique()
    if label_counts.max() != 1:
        raise RuntimeError("Identical images have conflicting labels")
    return result.drop_duplicates("image_hash").reset_index(drop=True)


def split_data(
    dataframe: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, remaining_df = train_test_split(
        dataframe,
        test_size=0.30,
        stratify=dataframe["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=0.50,
        stratify=remaining_df["label"],
        random_state=seed,
    )
    splits = tuple(frame.reset_index(drop=True) for frame in (train_df, val_df, test_df))
    hash_sets = [set(frame["image_hash"]) for frame in splits]
    if not (
        hash_sets[0].isdisjoint(hash_sets[1])
        and hash_sets[0].isdisjoint(hash_sets[2])
        and hash_sets[1].isdisjoint(hash_sets[2])
    ):
        raise RuntimeError("Exact image overlap detected between splits")
    return splits


def prepare_splits(
    dataset_path: str | Path, seed: int, verify: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    color_root = find_variant_root(dataset_path, "color")
    dataframe = build_image_table(color_root)
    if verify:
        verify_images(dataframe)
    dataframe = remove_exact_duplicates(dataframe)
    return split_data(dataframe, seed)


def attach_segmented_paths(
    dataframe: pd.DataFrame, dataset_path: str | Path
) -> pd.DataFrame:
    segmented_root = find_variant_root(dataset_path, "segmented")
    lookup = {}
    for class_name in APPLE_CLASSES:
        for path in (segmented_root / class_name).rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                # PlantVillage appends this suffix only to segmented images.
                # Remove it so the image can be paired with its colour version.
                source_stem = path.stem.removesuffix("_final_masked")
                key = (class_name, source_stem)
                if key in lookup:
                    raise RuntimeError(f"Duplicate segmented image key: {key}")
                lookup[key] = str(path)

    result = dataframe.copy()
    result["segmented_path"] = result.apply(
        lambda row: lookup.get((row["class_name"], Path(row["path"]).stem)), axis=1
    )
    missing = result["segmented_path"].isna().sum()
    if missing:
        raise RuntimeError(f"Could not match {missing} images to segmented versions")
    return result
