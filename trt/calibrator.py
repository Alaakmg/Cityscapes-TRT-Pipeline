"""INT8 entropy calibrator.

Calibration batches go through the same preprocessing as training
(segdeploy.data.preprocess_image). Calibration data that doesn't match
inference preprocessing silently ruins INT8 accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image

from segdeploy.data import preprocess_image


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        image_dir: str | Path,
        input_shape: tuple[int, int, int, int],
        n_images: int = 500,
        cache_file: str = "calibration.cache",
    ):
        super().__init__()
        self.input_shape = input_shape  # (N, 3, H, W), N assumed 1
        self.cache_file = Path(cache_file)

        self.paths: list[Path] = []
        if image_dir:
            self.paths = sorted(Path(image_dir).rglob("*.png"))[:n_images]
        self.index = 0
        self.device_input = cuda.mem_alloc(int(np.prod(input_shape)) * 4)

    def get_batch_size(self) -> int:
        return self.input_shape[0]

    def get_batch(self, names):
        if self.index >= len(self.paths):
            return None
        _, _, h, w = self.input_shape
        img = Image.open(self.paths[self.index])
        batch = preprocess_image(img, (h, w))[None].astype(np.float32)
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        self.index += 1
        if self.index % 50 == 0:
            print(f"  calibration: {self.index}/{len(self.paths)}")
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            print(f"Using calibration cache {self.cache_file}")
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache) -> None:
        self.cache_file.write_bytes(cache)
        print(f"Wrote calibration cache {self.cache_file}")
