# The $400 board vs the $2,000 GPU

*A 44M-parameter segmentation model, from FP32 PyTorch to fully-INT8
TensorRT on a Jetson Orin Nano. Code, raw numbers and decision log:
[github.com/Alaakmg/Cityscapes-TRT-Pipeline](https://github.com/Alaakmg/Cityscapes-TRT-Pipeline).*

My earlier Cityscapes segmentation projects were training projects: they end
at a checkpoint and a demo. This project is about everything after the
checkpoint: ONNX export, TensorRT engines, quantization,
and benchmarks on the hardware that would actually run the thing: a Jetson
Orin Nano Super, 8 GB, ~68 GB/s of memory bandwidth, the kind of compute
budget real robots get.

The model is deliberately boring: a ResNet50-encoder U-Net, 8 Cityscapes
categories, 512x1024 input, plain Conv/BN/ReLU and nearest upsampling so the
export never fights an opset. Trained to 0.8334 val mIoU in 65 minutes on a
rented RTX 5090 (~$1). The methodology was fixed before the first experiment:
accuracy is always measured on the deployed engine, never the source model;
batch 1; locked clocks; every number persisted as JSON with per-class IoU,
raw latencies and power.

## The desktop tells you a lie

First stop, the 5090:

| RTX 5090 | mIoU | latency |
|---|---|---|
| PyTorch FP32 | 0.8334 | 9.4 ms |
| TensorRT FP16 | 0.8334 | 2.5 ms |
| TensorRT INT8-PTQ | 0.8246 | 2.4 ms |

FP16 is a free lunch: 3.8x faster than eager PyTorch, half the size, mIoU
identical to four decimals. And INT8? It costs 0.9 mIoU points and buys about
4% of latency. On a card with 1.8 TB/s of bandwidth, this network is not
INT8-math-bound. If I had stopped here, the conclusion would have been
"skip INT8, ship FP16."

On the Jetson the same engines say the opposite:

| Jetson Orin Nano | mIoU | latency | energy/frame |
|---|---|---|---|
| TensorRT FP16 | 0.8334 | 38.9 ms | 632 mJ |
| TensorRT INT8-PTQ | 0.8236 | 20.4 ms | 261 mJ |

1.9x faster, 2.4x less energy. On a 68 GB/s device, halving the bytes moved
per layer is what the frame time responds to. Same model, same code, and the
decision flips. Only one of the two devices ships. A power-mode sweep
sharpened the rule: across 15 W, 25 W and MAXN modes, energy per frame is set
by the precision and barely moves with the mode; the mode only buys
frame rate. Battery
budget: pick the precision. Latency budget: pick the mode.

## Four engines to an honest QAT

Post-training quantization cost 0.9 mIoU points, so I fine-tuned with fake
quantization (NVIDIA ModelOpt) to recover it. The accuracy recovered on the
first try (0.8290). The *engine* was another story: slower than PTQ, on both
devices. `trtexec --dumpProfile` turned the mystery into a checklist, one
iteration per finding:

1. **v1**: 48 of the engine's layers ran in FP16. The profile showed the
   ResNet residual adds unfused: explicit quantization needs Q/DQ on the
   identity branch, which ModelOpt doesn't place. Added them: 48 -> 18 FP16
   layers, 31.6 -> 29.6 ms.
2. **v2 -> v3**: still unfused bottlenecks. The BatchNorm nodes between conv
   and add block TensorRT's conv+add+ReLU fusion (implicit PTQ never sees
   this; TensorRT folds BN itself before choosing precisions). Folding BN
   into the convs before quantization: 18 -> 2 FP16 layers, 26.4 ms. Side
   lesson: the BN-folded model diverges at the old fine-tuning LR; the
   normalization was doing stabilizing work.
3. **v4**: the last block of every encoder stage still emitted FP16, because
   its output also fed a decoder concat I had never quantized. Quantizers on
   the concat inputs: fully INT8, 19.9 ms, 0.8296 mIoU, faster than PTQ
   and 0.6 points more accurate.

Two tools made this trail safe to walk. A per-precision layer histogram,
persisted with every engine build, which caught a "QAT" engine that was
silently 100% FP16 (a wrapped export had renamed every tensor and invalidated
the calibration cache: no error, plausible numbers, wrong engine). And an
assertion on the quantizer count, which caught ModelOpt silently skipping
quantization entirely when the model already contains any quantizer. That
run trained an FP32 model and reported it as QAT. Quantization tooling fails
politely; the guards have to be impolite.

There was also a calibration cliff at the very start: ModelOpt's default
max-calibration met torchvision-ResNet's famous ~1300-magnitude BN activation
outliers and collapsed the model from 0.833 to 0.18 mIoU before training
began. Histogram calibration with percentile clipping, the same idea as
TensorRT's entropy calibrator, restored 0.819 with zero training.

## The decoder, not the backbone

The layer profile had one more thing to say: two thirds of the latency was in
the U-Net decoder, five blocks of 3x3 convs at high resolution, not in the
44M-parameter backbone everyone would instinctively shrink.

Latency depends on the architecture, not the weights. So I exported six
decoder variants *untrained*, measured them on the Jetson (probe engines land
within ~5% of trained ones), and only trained the three points the latency
curve made interesting. Two knobs: decoder width, and predicting at 1/2
resolution with a bilinear x2 on the logits.

| Jetson, INT8 | latency | mIoU | fps | energy/frame |
|---|---|---|---|---|
| baseline decoder | 20.4 ms | 0.8236 | 49 | 261 mJ |
| 0.5x width | 15.7 ms | 0.8240 | 64 | 167 mJ |
| 0.25x width, 1/2-res | **11.3 ms** | **0.8236** | **88** | **109 mJ** |

Halving the decoder costs 0.2 mIoU points. And the surprise: at 0.25x width,
the half-resolution variant is faster *and more accurate* than its
full-resolution twin, in both precisions. A starved full-res block is worse
than no full-res block. Neither FLOPs nor parameter counts predict any of
this (the half-res variants have *identical* parameter counts to their
full-res twins and are 14-19% faster). The final point: the baseline's INT8
accuracy at 1.8x its speed, 88 fps at 10 W, in a 29 MB engine.

## Takeaways

- **Measure on the target.** The desktop and the Jetson disagreed about INT8
  from the first table to the last. Every proxy (desktop latency, FLOPs,
  parameters) pointed somewhere wrong at least once in this project.
- **The profiler turns arguments into checklists.** All four QAT fixes and
  the decoder experiment came from `trtexec --dumpProfile` naming a specific
  layer.
- **Assert everything.** Three separate silent failures produced plausible
  numbers with wrong engines or wrong training. Persist the per-layer
  precisions of every engine you build and assert what you think you
  configured.
- Untrained latency probes are close to free: explore architectures on the
  device first, and train only the Pareto candidates.

Everything above is reproducible from the repo: the harness, the ONNX files,
the calibration caches, the raw JSONs behind every number, and an
[architecture decision log](adr.md) with the context and consequences of each
choice, including the ones that bit.
