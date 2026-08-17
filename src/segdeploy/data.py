"""Cityscapes dataset -> (image tensor, 8-category target).

Expects the official layout (registration required at cityscapes-dataset.com):

    <root>/leftImg8bit/{train,val}/<city>/*_leftImg8bit.png
    <root>/gtFine/{train,val}/<city>/*_gtFine_labelIds.png

Images are resized to a fixed (H, W) since the whole pipeline uses static
shapes, normalized with ImageNet statistics, and labels are mapped to the
8 top-level categories.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .labels import encode_labels

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img: Image.Image, size_hw: tuple[int, int]) -> np.ndarray:
    """PIL RGB image -> float32 CHW array, resized + ImageNet-normalized.

    Used by training, the parity check and the INT8 calibrator, so all of
    them see identical preprocessing.
    """
    h, w = size_hw
    img = img.convert("RGB").resize((w, h), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1).copy()


class CityscapesCategories(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        size_hw: tuple[int, int] = (512, 1024),
        augment: bool = False,
    ):
        self.root = Path(root)
        self.size_hw = size_hw
        self.augment = augment

        img_dir = self.root / "leftImg8bit" / split
        self.images = sorted(img_dir.rglob("*_leftImg8bit.png"))
        if not self.images:
            raise FileNotFoundError(f"No images under {img_dir}")
        self.targets = [
            Path(str(p).replace("leftImg8bit", "gtFine").replace("_gtFine.png", "_gtFine_labelIds.png"))
            for p in self.images
        ]
        # The replace above rewrites both the directory and the file suffix;
        # fix the suffix explicitly to be robust.
        self.targets = [
            t.with_name(t.name.replace("_gtFine_labelIds_labelIds", "_gtFine_labelIds"))
            for t in self.targets
        ]
        missing = [t for t in self.targets[:5] if not t.exists()]
        if missing:
            raise FileNotFoundError(f"Label file not found, e.g. {missing[0]}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.images[i])
        lbl = Image.open(self.targets[i])

        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)

        x = preprocess_image(img, self.size_hw)
        h, w = self.size_hw
        lbl = lbl.resize((w, h), Image.NEAREST)
        y = encode_labels(np.asarray(lbl))

        return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))
