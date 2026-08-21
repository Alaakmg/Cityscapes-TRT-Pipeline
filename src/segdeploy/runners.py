"""Unified inference runners.

Every backend exposes the same interface:

    runner(x: np.ndarray[N,3,H,W] float32) -> np.ndarray[N,C,H,W] float32 (logits)

so evaluate.py and benchmark.py don't care which backend they are running.

TensorRT imports are guarded: TrtRunner only works on a machine with the
TensorRT Python bindings (e.g. a Jetson with JetPack).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import torch


class TorchRunner:
    def __init__(self, model: torch.nn.Module, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x).to(self.device)
        return self.model(t).float().cpu().numpy()

    def synchronize(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()


class OnnxRunner:
    def __init__(self, onnx_path: str | Path, providers: list[str] | None = None):
        import onnxruntime as ort

        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x})[0]

    def synchronize(self) -> None:  # ORT run() is synchronous
        pass


class TrtRunner:
    """Minimal TensorRT execution wrapper (TensorRT >= 8.5 tensor-name API).

    Assumes a single input binding and a single output binding, which is what
    `export/export_onnx.py` + `trt/build_engine.py` produce. The output may be
    logits (N, C, H, W) float or, for engines exported with `--argmax`, a class
    mask (N, H, W) int32.

    Device buffers are torch CUDA tensors and host buffers are pinned torch
    tensors: torch is present on every machine that runs our engines anyway,
    this avoids a pycuda dependency, and pinned host memory turns the H2D/D2H
    copies into true async DMA instead of staged pageable copies (measured on
    Jetson Orin Nano: ~2.8 ms of overhead per frame with pageable numpy
    buffers, see docs/findings.md).
    """

    _DTYPES: ClassVar[dict[str, torch.dtype]] = {
        "FLOAT": torch.float32,
        "HALF": torch.float16,
        "INT32": torch.int32,
        "INT8": torch.int8,
        "UINT8": torch.uint8,
        "BOOL": torch.bool,
    }

    def __init__(self, engine_path: str | Path):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name).name == "INPUT":
                self.input_name = name
            else:
                self.output_name = name
        assert self.input_name and self.output_name

        self.in_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.out_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.out_dtype = self._DTYPES[self.engine.get_tensor_dtype(self.output_name).name]

        self.stream = torch.cuda.Stream()
        self.h_in = torch.empty(self.in_shape, dtype=torch.float32, pin_memory=True)
        self.h_out = torch.empty(self.out_shape, dtype=self.out_dtype, pin_memory=True)
        self.d_in = torch.empty(self.in_shape, dtype=torch.float32, device="cuda")
        self.d_out = torch.empty(self.out_shape, dtype=self.out_dtype, device="cuda")
        self.context.set_tensor_address(self.input_name, self.d_in.data_ptr())
        self.context.set_tensor_address(self.output_name, self.d_out.data_ptr())

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Returns a view of the pinned output buffer, valid until the next call."""
        assert tuple(x.shape) == self.in_shape, f"expected {self.in_shape}, got {x.shape}"
        self.h_in.numpy()[...] = x
        with torch.cuda.stream(self.stream):
            self.d_in.copy_(self.h_in, non_blocking=True)
            ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            assert ok, "TensorRT execution failed"
            self.h_out.copy_(self.d_out, non_blocking=True)
        self.stream.synchronize()
        return self.h_out.numpy()

    def synchronize(self) -> None:
        self.stream.synchronize()
