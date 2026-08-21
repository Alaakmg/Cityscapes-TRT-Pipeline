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
| PyTorch | FP32 | RTX 5090 | 0.8334 | – | 9.35 | 9.66 | 107 | 176 |
| TensorRT | FP32 | RTX 5090 | 0.8334 | ±0.0000 | 6.17 | 6.19 | 162 | 251 |
| TensorRT | FP16 | RTX 5090 | 0.8334 | −0.0000 | 2.96 | 2.97 | 338 | 89 |
| TensorRT | INT8 (PTQ) | RTX 5090 | 0.8246 | −0.0088 | 2.94 | 2.96 | 340 | 46 |
| TensorRT | INT8 (QAT) | RTX 5090 | 0.8290 | −0.0044 | 3.17 | 3.18 | 316 | 107 |
| TensorRT | FP32 | Jetson Orin Nano | 0.8334 | ±0.0000 | 111.7 | 112.5 | 9.0 | 176 |
| TensorRT | FP16 | Jetson Orin Nano | 0.8334 | −0.0000 | 41.6 | 42.4 | 24.1 | 88 |
| TensorRT | INT8 (PTQ) | Jetson Orin Nano | 0.8236 | −0.0098 | **22.9** | 23.6 | **43.6** | 45 |
| TensorRT | INT8 (QAT) | Jetson Orin Nano | 0.8290 | −0.0044 | 34.3 | 35.0 | 29.1 | 45 |

FP16 so far: **3.2x faster than eager PyTorch, 2.8x smaller, zero measurable mIoU loss**
(per-class IoU stable to 4 decimals, including the thin-structure classes). ONNX export
parity: max abs logit diff 4.1e-05, argmax mismatch 3.8e-06 (`results/.../parity_onnx.json`).

Jetson: **INT8-PTQ is 1.8x faster than FP16 and 2.4x more energy-efficient**
(268 vs 632 mJ/frame) for −1.0 mIoU point. Same model, opposite conclusion from the
desktop — measured on the target, not extrapolated. Details and the profiling
breakdown in [`docs/findings.md`](docs/findings.md).

INT8-PTQ observations (desktop): −0.9 mIoU points for a 2x smaller engine, but **no
latency win over FP16 at batch 1 on the 5090** — the workload isn't INT8-math-bound
there. The INT8 case rests on the bandwidth-starved Jetson, which is the point of
measuring on the target. Degradation is spread across classes (construction/object/
nature −1.4 to −1.6 pts), the QAT row exists to claw that back.

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
- [ ] C++ inference wrapper (`cpp/`), write-up

## Dataset

Cityscapes requires registration (research/non-commercial terms):
https://www.cityscapes-dataset.com. This repo contains no dataset content, point
`data_root` at your local copy.
