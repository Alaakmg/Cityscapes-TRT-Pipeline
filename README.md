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

| Variant | Precision | Device | mIoU | Δ vs FP32 | Latency mean (ms) | p95 (ms) | img/s | Size (MB) |
|---|---|---|---|---|---|---|---|---|
| PyTorch | FP32 | RTX (desktop) | – | – | – | – | – | – |
| ONNX Runtime | FP32 | CPU / CUDA EP | – | – | – | – | – | – |
| TensorRT | FP32 | desktop | – | – | – | – | – | – |
| TensorRT | FP16 | desktop | – | – | – | – | – | – |
| TensorRT | INT8 (PTQ) | desktop | – | – | – | – | – | – |
| TensorRT | INT8 (QAT) | desktop | – | – | – | – | – | – |
| TensorRT | FP16 | Jetson Orin Nano | – | – | – | – | – | – |
| TensorRT | INT8 (QAT) | Jetson Orin Nano | – | – | – | – | – | – |

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
- [ ] FP32 baseline trained on Cityscapes (8 categories)
- [ ] ONNX export + numerical parity gate
- [ ] TensorRT FP32/FP16, accuracy measured on the engine itself
- [ ] INT8: PTQ (entropy calibration), then QAT (`pytorch-quantization`, Q/DQ export)
- [ ] Jetson Orin Nano: on-device benchmarks, `trtexec` / Nsight profiling
- [ ] C++ inference wrapper (`cpp/`), write-up

## Dataset

Cityscapes requires registration (research/non-commercial terms):
https://www.cityscapes-dataset.com. This repo contains no dataset content, point
`data_root` at your local copy.
