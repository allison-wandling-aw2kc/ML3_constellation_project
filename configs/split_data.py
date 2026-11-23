#!/usr/bin/env python3
import random
import shutil
from pathlib import Path


def split_dataset_by_name(train_image_dir, train_label_dir, val_ratio=0.15, seed=42):
    """Move a seeded 15% subset of train images and matching labels into val."""
    train_image_dir = Path(train_image_dir)
    train_label_dir = Path(train_label_dir)

    if not train_image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {train_image_dir}")
    if not train_label_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {train_label_dir}")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    images_root = train_image_dir.parent
    labels_root = train_label_dir.parent
    val_image_dir = images_root / "val"
    val_label_dir = labels_root / "val"

    val_image_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(p for p in train_image_dir.iterdir() if p.is_file())
    if not image_files:
        raise ValueError(f"No image files found in {train_image_dir}")

    existing_val_count = len(list(val_image_dir.glob('*.*')))
    total_count = len(image_files) + existing_val_count
    target_val_count = max(1, int(total_count * val_ratio))
    remaining_to_move = max(0, target_val_count - existing_val_count)

    random.seed(seed)
    random.shuffle(image_files)

    val_images = image_files[:remaining_to_move]

    moved = 0
    for image_path in val_images:
        label_path = train_label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(
                f"Matching label not found for image: {image_path.name} -> {label_path}"
            )

        image_destination = val_image_dir / image_path.name
        label_destination = val_label_dir / label_path.name

        if image_destination.exists() or label_destination.exists():
            continue

        shutil.move(str(image_path), str(image_destination))
        shutil.move(str(label_path), str(label_destination))
        moved += 1

    print(f"Moved {moved} additional image/label pairs from train to val")
    print(f"Images: {val_image_dir}")
    print(f"Labels: {val_label_dir}")
    print(f"Remaining training images: {len(list(train_image_dir.glob('*')))}")


if __name__ == "__main__":
    dataset_root = Path(__file__).resolve().parent / "constellation_dataset"
    train_image_dir = dataset_root / "images" / "train"
    train_label_dir = dataset_root / "labels" / "train"
    split_dataset_by_name(train_image_dir, train_label_dir, val_ratio=0.15, seed=42)
