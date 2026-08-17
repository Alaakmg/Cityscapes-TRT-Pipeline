"""Cityscapes label handling.

Maps the 34 raw Cityscapes label ids (found in *_labelIds.png) to the 8
top-level categories used throughout this project:

    0 void | 1 flat | 2 construction | 3 object | 4 nature | 5 sky | 6 human | 7 vehicle

Mirrors the official `category` field of cityscapesscripts.helpers.labels.
"""

from __future__ import annotations

import numpy as np
import torch

NUM_CLASSES = 8

CATEGORY_NAMES = [
    "void",
    "flat",
    "construction",
    "object",
    "nature",
    "sky",
    "human",
    "vehicle",
]

# Raw labelId -> category id, following the official Cityscapes label table.
_ID_TO_CAT = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,          # void group
    7: 1, 8: 1, 9: 1, 10: 1,                            # flat: road, sidewalk, parking, rail track
    11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2,           # construction
    17: 3, 18: 3, 19: 3, 20: 3,                         # object: pole(s), traffic light/sign
    21: 4, 22: 4,                                       # nature: vegetation, terrain
    23: 5,                                              # sky
    24: 6, 25: 6,                                       # human: person, rider
    26: 7, 27: 7, 28: 7, 29: 7, 30: 7, 31: 7, 32: 7, 33: 7,  # vehicle
}

# Vectorized lookup table. Index 34 is used for the raw id -1 (license plate),
# which we fold into 'vehicle'.
ID_TO_CAT_LUT = np.zeros(35, dtype=np.uint8)
for raw_id, cat in _ID_TO_CAT.items():
    ID_TO_CAT_LUT[raw_id] = cat
ID_TO_CAT_LUT[34] = 7

# Muted display palette for qualitative results (RGB).
PALETTE = np.array(
    [
        [0, 0, 0],        # void
        [128, 64, 128],   # flat
        [70, 70, 70],     # construction
        [220, 220, 0],    # object
        [107, 142, 35],   # nature
        [70, 130, 180],   # sky
        [220, 20, 60],    # human
        [0, 0, 142],      # vehicle
    ],
    dtype=np.uint8,
)


def encode_labels(raw: np.ndarray) -> np.ndarray:
    """Map a raw labelIds array (H, W) to category ids (H, W) in [0, 7]."""
    raw = raw.astype(np.int64)
    raw = np.where(raw < 0, 34, raw)
    if raw.max() > 34:
        raise ValueError(f"Unexpected raw label id {raw.max()} (valid range: -1..33)")
    return ID_TO_CAT_LUT[raw]


def colorize(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    """Category mask (H, W) -> RGB image (H, W, 3) for visualization."""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    return PALETTE[mask.astype(np.int64)]
