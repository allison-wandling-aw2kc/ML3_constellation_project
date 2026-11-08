#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path


def split_dataset_by_name(image_dir, label_dir, output_root=None, val_ratio=0.5, seed=42):
    """
    Split images and labels in lockstep using the same randomized selection.

    Output structure:
      <images_root>/val_split/
      <images_root>/test_split/
      <labels_root>/val_split/
      <labels_root>/test_split/
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    images_root = image_dir.parent
    labels_root = label_dir.parent

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    image_files = sorted([p for p in image_dir.iterdir() if p.is_file()])
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")

    random.seed(seed)
    random.shuffle(image_files)

    split_index = max(1, int(len(image_files) * val_ratio))
    val_images = image_files[:split_index]
    test_images = image_files[split_index:]

    val_image_out = images_root / "val_split"
    val_label_out = labels_root / "val_split"
    test_image_out = images_root / "test_split"
    test_label_out = labels_root / "test_split"

    for folder in [val_image_out, val_label_out, test_image_out, test_label_out]:
        folder.mkdir(parents=True, exist_ok=True)

    def copy_matching_label(image_path, out_img_dir, out_label_dir):
        stem = image_path.stem
        label_path = label_dir / f"{stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Matching label not found for image: {image_path.name} -> {label_path}")

        shutil.copy2(image_path, out_img_dir / image_path.name)
        shutil.copy2(label_path, out_label_dir / label_path.name)

    for img in val_images:
        copy_matching_label(img, val_image_out, val_label_out)

    for img in test_images:
        copy_matching_label(img, test_image_out, test_label_out)

    print(f"Copied {len(val_images)} image/label pairs to {val_image_out} and {val_label_out}")
    print(f"Copied {len(test_images)} image/label pairs to {test_image_out} and {test_label_out}")
    print(f"Total source image files: {len(image_files)}")


if __name__ == "__main__":
    image_dir = "/Users/alliewandling/Documents/uva_masters/ML3/constellation-dataset/configs/constellation_dataset/images/val"
    label_dir = "/Users/alliewandling/Documents/uva_masters/ML3/constellation-dataset/configs/constellation_dataset/labels/val"
    output_root = "/Users/alliewandling/Documents/uva_masters/ML3/constellation-dataset/configs/constellation_dataset"
    split_dataset_by_name(image_dir, label_dir, output_root, val_ratio=0.5, seed=42)
