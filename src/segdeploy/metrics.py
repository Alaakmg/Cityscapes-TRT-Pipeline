"""Streaming confusion-matrix metrics, shared by all backends (torch/onnx/trt)."""

from __future__ import annotations

import numpy as np


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, target: np.ndarray, pred: np.ndarray) -> None:
        """target/pred: integer arrays of identical shape with values in [0, C)."""
        t = target.reshape(-1).astype(np.int64)
        p = pred.reshape(-1).astype(np.int64)
        valid = (t >= 0) & (t < self.num_classes)
        idx = self.num_classes * t[valid] + p[valid]
        self.mat += np.bincount(idx, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def iou_per_class(self) -> np.ndarray:
        tp = np.diag(self.mat).astype(np.float64)
        fp = self.mat.sum(axis=0) - tp
        fn = self.mat.sum(axis=1) - tp
        denom = tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(denom > 0, tp / denom, np.nan)
        return iou

    def miou(self) -> float:
        return float(np.nanmean(self.iou_per_class()))

    def pixel_accuracy(self) -> float:
        return float(np.diag(self.mat).sum() / max(self.mat.sum(), 1))

    def summary(self, class_names: list[str] | None = None) -> str:
        iou = self.iou_per_class()
        names = class_names or [str(i) for i in range(self.num_classes)]
        lines = [f"{n:>14s}: {v:.4f}" for n, v in zip(names, iou)]
        lines.append(f"{'mIoU':>14s}: {self.miou():.4f}")
        lines.append(f"{'pixel acc':>14s}: {self.pixel_accuracy():.4f}")
        return "\n".join(lines)
