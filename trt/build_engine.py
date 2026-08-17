"""Build a TensorRT engine from the ONNX model.

Run ON THE TARGET DEVICE: engines are specific to the GPU architecture and
TensorRT version that built them, an engine built on a desktop GPU will not
run on a Jetson.

    # FP16
    python trt/build_engine.py --onnx model.onnx --out model_fp16.engine --fp16

    # INT8 with entropy calibration on Cityscapes train images
    python trt/build_engine.py --onnx model.onnx --out model_int8.engine --int8 \
        --calib-dir /data/cityscapes/leftImg8bit/train --calib-images 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tensorrt as trt


def build(args) -> None:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = Path(args.onnx).read_bytes()
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)

    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if args.int8:
        # Deferred import: calibration uses IInt8EntropyCalibrator2, which
        # TensorRT 11 removed (implicit INT8 quantization is gone there, only
        # explicit Q/DQ remains). We pin TRT 10.x, matching JetPack on the
        # Jetson target; the import stays lazy so FP32/FP16 builds don't care.
        from calibrator import EntropyCalibrator  # local module, same directory

        config.set_flag(trt.BuilderFlag.INT8)
        # FP16 fallback for layers TensorRT declines to run in INT8.
        config.set_flag(trt.BuilderFlag.FP16)
        input_shape = tuple(network.get_input(0).shape)  # (1, 3, H, W)
        config.int8_calibrator = EntropyCalibrator(
            image_dir=args.calib_dir,
            input_shape=input_shape,
            n_images=args.calib_images,
            cache_file=args.calib_cache,
        )

    print("Building engine (this can take several minutes)...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.time() - t0
    if serialized is None:
        raise RuntimeError("Engine build failed")
    # IHostMemory supports the buffer protocol but (as of TRT 10.16) not len()
    size_mb = memoryview(serialized).nbytes / 1e6
    Path(args.out).write_bytes(serialized)
    print(f"Saved {args.out} ({size_mb:.1f} MB)")

    # engines are disposable/device-specific, keep the build facts next to them
    meta = {
        "onnx": args.onnx,
        "engine": args.out,
        "fp16": args.fp16,
        "int8": args.int8,
        "calib_images": args.calib_images if args.int8 else None,
        "engine_mb": round(size_mb, 2),
        "build_seconds": round(build_s, 1),
        "trt_version": trt.__version__,
    }
    meta_path = args.out + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib-dir", help="Directory of calibration images (INT8)")
    ap.add_argument("--calib-images", type=int, default=500)
    ap.add_argument("--calib-cache", default="calibration.cache")
    ap.add_argument("--workspace-gb", type=int, default=2)
    args = ap.parse_args()

    if args.int8 and not args.calib_dir and not Path(args.calib_cache).exists():
        ap.error("--int8 requires --calib-dir (or an existing --calib-cache)")
    build(args)


if __name__ == "__main__":
    main()
