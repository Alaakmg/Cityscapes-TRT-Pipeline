# cityscapes-trt-pipeline

Deployment pipeline for a Cityscapes semantic segmentation model (ResNet50 U-Net,
8 categories): PyTorch -> ONNX -> TensorRT (FP32 / FP16 / INT8-PTQ / INT8-QAT),
targeting a Jetson Orin Nano.

Follow-up to my earlier Cityscapes training projects (U-Net / U-Net3+ in Keras).
This one is about everything that happens after training: export, quantization,
benchmarking, profiling.

## Results

Accuracy is always measured on the deployed artifact itself
(`python -m segdeploy.evaluate --backend ...`), never on the source PyTorch model.
Latency: batch 1, 512x1024, 20 warmup + 200 timed iterations, explicit device sync.

Desktop = RTX 5090 (RunPod), TensorRT 10.16.1, torch 2.8.0+cu128.
Jetson = Orin Nano Super 8 GB, JetPack 7.2.1 (L4T R39.2.1, TensorRT 10.16.2, CUDA 13.2),
power mode MAXN_SUPER with `jetson_clocks` (GPU locked 1.02 GHz, EMC 3.2 GHz).
Jetson power = module input (`VDD_IN`) sampled by `tegrastats` at 2 Hz during the benchmark.
Raw JSONs, tegrastats logs and `trtexec` layer profiles in `results/`.

| Variant | Precision | Device | mIoU | Δ vs FP32 | Latency mean (ms) | p95 (ms) | img/s | Size (MB) |
|---|---|---|---|---|---|---|---|---|
| PyTorch | FP32 | RTX 5090 | 0.8334 | – | 9.41 | 9.54 | 106 | 176 |
| TensorRT | FP32 | RTX 5090 | 0.8334 | ±0.0000 | 6.13 | 6.30 | 163 | 251 |
| TensorRT | FP16 | RTX 5090 | 0.8334 | −0.0000 | 2.48 | 2.59 | 403 | 89 |
| TensorRT | INT8 (PTQ) | RTX 5090 | 0.8246 | −0.0088 | 2.39 | 2.50 | 419 | 46 |
| TensorRT | INT8 (QAT v1) | RTX 5090 | 0.8290 | −0.0044 | 2.54 | 2.60 | 394 | 107 |
| TensorRT | INT8 (QAT v2) | RTX 5090 | 0.8287 | −0.0047 | 2.50 | 2.51 | 400 | 107 |
| TensorRT | FP32 | Jetson Orin Nano | 0.8334 | ±0.0000 | 108.8 | 109.1 | 9.2 | 176 |
| TensorRT | FP16 | Jetson Orin Nano | 0.8334 | −0.0000 | 38.9 | 39.0 | 25.7 | 88 |
| TensorRT | INT8 (PTQ) | Jetson Orin Nano | 0.8236 | −0.0098 | **20.4** | 20.5 | **49.0** | 45 |
| TensorRT | INT8 (QAT v1) | Jetson Orin Nano | 0.8290 | −0.0044 | 31.8 | 31.8 | 31.5 | 45 |
| TensorRT | INT8 (QAT v2) | Jetson Orin Nano | – | – | – | – | – | – |

All rows: harness v2 (pinned host buffers, private CUDA stream), within ~0.5 ms of `trtexec`.
The original v1-harness numbers stay in `results/` for reference. QAT v2 = residual
branches quantized so TensorRT can fuse the ResNet bottlenecks; Jetson row pending.

FP16 on the desktop: 3.8x faster than eager PyTorch, 2.8x smaller, zero measurable mIoU loss
(per-class IoU stable to 4 decimals, including the thin-structure classes). ONNX export
parity: max abs logit diff 4.1e-05, argmax mismatch 3.8e-06 (`results/.../parity_onnx.json`).

Jetson: INT8-PTQ is 1.8x faster than FP16 and 2.4x more energy-efficient
(268 vs 632 mJ/frame) for −1.0 mIoU point. The desktop table alone argues
the other way; the Jetson is the device that ships. Details and the profiling
breakdown in [`docs/findings.md`](docs/findings.md).

Power-mode sweep (all engines, 15 W / 25 W / MAXN_SUPER, clocks locked and DVFS):
energy per frame is set by the precision, the power mode only moves latency.
INT8-PTQ is the only configuration above 30 fps at 25 W.

![jetson power modes](docs/jetson_power_modes.png)

| Jetson, clocks locked | 15 W | 25 W | MAXN_SUPER |
|---|---|---|---|
| FP16 ms / fps / W / mJ per frame | 64.9 / 15.4 / 9.9 / 645 | 44.2 / 22.6 / 13.7 / 606 | 41.5 / 24.1 / 15.0 / 621 |
| INT8-PTQ ms / fps / W / mJ per frame | 36.0 / 27.8 / 7.9 / 283 | 25.0 / 40.0 / 10.3 / 258 | 22.9 / 43.6 / 11.4 / 261 |
| INT8 vs FP16 | 1.80x faster, 2.3x less energy | 1.77x, 2.3x | 1.81x, 2.4x |

INT8-PTQ observations (desktop): −0.9 mIoU points for a 2x smaller engine, but only a
4% latency win over FP16 at batch 1 on the 5090. The workload isn't INT8-math-bound
there; the INT8 case rests on the bandwidth-starved Jetson. Degradation is spread
across classes (construction/object/nature −1.4 to −1.6 pts), the QAT row exists
to win that back.

Per-class IoU breakdowns will go in `docs/findings.md`. Thin structures (poles,
humans) are usually the first classes to suffer under quantization, so I track
per-class IoU everywhere, not just the mean.

## Pipeline

```
train.py ──> best.pt ──> export_onnx.py ──> model.onnx ──> build_engine.py ──> .engine
                │              │                                  │
                │              └── check_parity.py (CI gate)      ├── FP16
                │                                                 ├── INT8 PTQ (entropy calib.)
                └── qat_finetune.py ──> Q/DQ ONNX ────────────────└── INT8 QAT
```

`segdeploy/runners.py` gives torch / ONNX Runtime / TensorRT the same interface,
so evaluation and benchmarking run the exact same code for every backend.

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q

# FP32 baseline (needs Cityscapes at data_root, see configs/baseline.yaml)
python -m segdeploy.train --config configs/baseline.yaml

# export + parity gate
python export/export_onnx.py --checkpoint runs/fp32/best.pt --out model.onnx
python export/check_parity.py --checkpoint runs/fp32/best.pt --onnx model.onnx

# engines (run ON the target device, engines are not portable)
python trt/build_engine.py --onnx model.onnx --out model_fp16.engine --fp16
python trt/build_engine.py --onnx model.onnx --out model_int8.engine --int8 \
    --calib-dir $CITYSCAPES/leftImg8bit/train --calib-images 500

# INT8 QAT: fine-tune with fake quant (modelopt), export Q/DQ onnx,
# then build --int8 with no calibrator (ranges live in the graph)
python trt/qat_finetune.py --config configs/baseline.yaml --checkpoint runs/fp32/best.pt --out runs/qat
python trt/build_engine.py --onnx runs/qat/model_qat.onnx --out model_int8_qat.engine --int8

# accuracy + latency of any artifact
python -m segdeploy.evaluate  --backend trt --model model_int8.engine --data-root $CITYSCAPES
python -m segdeploy.benchmark --backend trt --model model_int8.engine
```

## Outputs

Every stage writes its numbers to disk, nothing lives only in stdout:

| Stage | Writes | Used for |
|---|---|---|
| `train.py` / `qat_finetune.py` | `<out_dir>/metrics.jsonl` (per-step loss/LR, per-epoch val mIoU + per-class IoU) | loss/mIoU curves, baseline vs QAT overlays |
| `evaluate.py --out r.json` | mIoU, pixel acc, per-class IoU, full confusion matrix | confusion heatmaps, per-class degradation bars |
| `benchmark.py --out b.json` | summary stats + raw per-iteration latencies | latency histograms, p99 / tail analysis |
| `check_parity.py --out p.json` | max abs/rel diff, argmax mismatch, pass flag | export fidelity across opsets/versions |
| `build_engine.py` | `<engine>.meta.json` (precision, size, build time, TRT version) | size vs precision, engine provenance |

One directory per variant (`results/trt_fp16/`, `results/trt_int8_qat/`, ...).
Engines are disposable, their `.meta.json` sidecars are not.

Training curves come from:

```bash
python scripts/plot_training.py --run runs/fp32 --out docs/curves_fp32.png
# overlay runs, e.g. baseline vs QAT:
python scripts/plot_training.py --run runs/fp32 --run runs/qat --out docs/curves_qat.png
```

## Design notes

The full decision log, with context and consequences for each choice, is in
[`docs/adr.md`](docs/adr.md). Short version:

- **Static input shape (1x3x512x1024).** Cameras run at a fixed resolution anyway,
  and static shapes keep TensorRT profiles and INT8 calibration simple. The cost is
  one engine per resolution, which I can live with.
- **Plain Conv/BN/ReLU + nearest upsampling, single output head.** No deformable
  convs, no custom ops. I'd rather spend time on quantization than on opset fights.
- **One preprocessing function** (`segdeploy.data.preprocess_image`) shared by
  training, parity checks and the INT8 calibrator. Calibration data that doesn't
  match inference preprocessing silently ruins INT8 accuracy.
- **Engines are built on the device that runs them.** TensorRT engines are tied to
  GPU architecture + TRT version. The repo ships ONNX, never engines.

## Roadmap

- [x] skeleton, CI, benchmark methodology fixed before any experiment
- [x] FP32 baseline trained on Cityscapes (8 categories): val mIoU 0.833, 60 epochs
- [x] ONNX export + numerical parity gate
- [x] TensorRT FP32/FP16, accuracy measured on the engine itself
- [x] INT8: PTQ (entropy calibration), then QAT (modelopt, Q/DQ export)
- [x] Jetson Orin Nano: on-device benchmarks, power, `trtexec` layer profiling

## Dataset

Cityscapes requires registration (research/non-commercial terms):
https://www.cityscapes-dataset.com. This repo contains no dataset content, point
`data_root` at your local copy.
