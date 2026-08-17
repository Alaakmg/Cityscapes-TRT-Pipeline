"""Latency/throughput harness shared by all backends.

Methodology:
  * batch size 1 unless overridden (the realistic setting for a camera feed)
  * fixed warmup iterations before timing
  * per-iteration wall-clock timing with explicit device synchronization
  * reports mean, p50, p95 latency and images/sec

Usage:
    python -m segdeploy.benchmark --backend torch --checkpoint runs/fp32/best.pt
    python -m segdeploy.benchmark --backend onnx  --model model.onnx
    python -m segdeploy.benchmark --backend trt   --model model_fp16.engine
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def run_benchmark(runner, shape=(1, 3, 512, 1024), warmup=20, iters=200) -> dict:
    x = np.random.rand(*shape).astype(np.float32)

    for _ in range(warmup):
        runner(x)
    runner.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        runner(x)
        runner.synchronize()
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000.0
    return {
        "shape": list(shape),
        "iters": iters,
        "latency_ms_mean": float(times_ms.mean()),
        "latency_ms_p50": float(np.percentile(times_ms, 50)),
        "latency_ms_p95": float(np.percentile(times_ms, 95)),
        "throughput_img_s": float(shape[0] * iters / (times_ms.sum() / 1000.0)),
        # raw per-iteration latencies, so histograms/p99/tails can be
        # plotted later without re-running on the same hardware
        "latency_ms_raw": [round(float(t), 4) for t in times_ms],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "onnx", "trt"], required=True)
    ap.add_argument("--checkpoint", help="PyTorch checkpoint (backend=torch)")
    ap.add_argument("--model", help="ONNX file or TRT engine path")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", help="Optional JSON output path")
    args = ap.parse_args()

    if args.backend == "torch":
        import torch

        from .model import build_model
        from .runners import TorchRunner

        model = build_model(pretrained=False)
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(state.get("model", state))
        runner = TorchRunner(model)
    elif args.backend == "onnx":
        from .runners import OnnxRunner

        runner = OnnxRunner(args.model)
    else:
        from .runners import TrtRunner

        runner = TrtRunner(args.model)

    stats = run_benchmark(
        runner, shape=(args.batch, 3, args.height, args.width), iters=args.iters
    )
    print(json.dumps(stats, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
