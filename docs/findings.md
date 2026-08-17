# Findings

Working notes as the pipeline progresses. Numbers come from the JSONs in
`results/` — nothing here is quoted from memory.

## FP32 baseline (RTX 5090, 60 epochs, ~65 min, ~$1)

Best val mIoU **0.8334** at epoch 35; the last 25 epochs only flatten the curve
(final epoch 0.8310). My old TF/Keras U-Net on the same 8-category task topped
out at 0.786 with 60.5M params — this one does +4.7 points with 44M, mostly
thanks to the ImageNet-pretrained ResNet50 encoder (epoch 1 already hits 0.748)
and a properly scheduled AdamW + cosine run.

![training curves](curves_fp32.png)

Per-class IoU makes the difficulty ranking obvious, and it never changes across
backends or precisions:

| | flat | sky | vehicle | nature | construction | human | void | object |
|---|---|---|---|---|---|---|---|---|
| FP32 | 0.937 | 0.927 | 0.912 | 0.908 | 0.881 | 0.774 | 0.713 | 0.616 |

`object` (poles, traffic signs — thin structures) and `human` are the weak
classes. They are the ones to watch under quantization.

## ONNX export is exact (for practical purposes)

Max abs logit diff vs PyTorch: **4.1e-05**; argmax flips: **3.8e-06** of pixels
(~8 per million). One trap for the unwary: max *relative* diff looks alarming
(~2.3) but it lives entirely on near-zero logits where relative error is
meaningless — gate on absolute diff + argmax agreement, not relative.

## FP16 is a free lunch here

| backend | mIoU | mean ms | p95 ms | img/s | MB |
|---|---|---|---|---|---|
| PyTorch FP32 | 0.8334 | 9.35 | 9.66 | 107 | 176 |
| TRT FP32 | 0.8334 | 6.17 | 6.19 | 162 | 251 |
| TRT FP16 | 0.8334 | 2.96 | 2.97 | 338 | 89 |
| TRT INT8-PTQ | 0.8246 | 2.94 | 2.96 | 340 | 46 |

FP16: 3.2x over eager PyTorch, 2.8x smaller, and mIoU identical to four
decimals — including `object` and `human`. Also worth noticing: TensorRT's
latency distribution is *tight* (FP32: 6.17 mean / 6.19 p95) where PyTorch
eager has visible spread. Determinism is its own feature on an AV stack.

## INT8 does NOT beat FP16 on a desktop GPU at batch 1

The surprise of the table: INT8-PTQ is 2.94 ms vs FP16's 2.96 ms — nothing.
At batch 1 on a 5090 (1.8 TB/s of bandwidth) this network is not INT8-math
bound; halving the weights (46 MB engine) doesn't move the needle when
activations dominate traffic and the tensor cores were already underfed at
FP16. The INT8 story has to be earned on the bandwidth-starved target (Jetson
Orin Nano, ~68 GB/s) — which is exactly why the plan measures there and
doesn't extrapolate from desktop numbers.

Meanwhile INT8-PTQ costs **−0.9 mIoU points**, spread across classes
(construction/object/nature each lose 1.4–1.6 pts) rather than concentrated in
the thin classes as I expected. QAT's job is to claw that back; if it recovers
most of the 0.9 while keeping the 46 MB engine, INT8 becomes strictly better
than FP16 *on the Jetson* or it doesn't ship. That's the decision the numbers
have to make.

## Toolchain notes (the parts nobody's blog post mentions)

- **TensorRT 11 removed `IInt8EntropyCalibrator2`** — implicit INT8
  quantization is gone there; explicit Q/DQ is the only INT8 path. I pinned
  desktop TRT to 10.x to match what JetPack ships on the Orin, which also
  keeps the entropy-calibration PTQ row reproducible. Practical consequence:
  calibrator-style PTQ is legacy API on borrowed time; the QAT/Q-DQ pipeline
  is the one with a future.
- **pycuda is a liability as a dependency** — no wheels, needs nvcc + boost
  headers at install time, and the RunPod PyTorch image ships neither. Torch
  is a hard dependency of this project anyway, so the TRT runner and the
  calibrator now use torch CUDA tensors as device buffers (`data_ptr()` +
  current stream). One less install step on every target, Jetson included.
- **`IHostMemory` lost `len()` somewhere before TRT 10.16** — use
  `memoryview(x).nbytes`. Small, but it broke the engine-metadata sidecar.
- Engine builds look slow (~minutes) only on the first build per machine —
  TensorRT's kernel-timing cache makes every subsequent build take seconds.
  Consequence: rebuilding all engines from ONNX on a fresh pod costs almost
  nothing, so engines really can be treated as disposable.

## Next

- QAT fine-tune (modelopt, Q/DQ export) — target: recover most of the −0.9.
- Jetson Orin Nano: same table on the real target, `tegrastats` power numbers,
  `trtexec`/Nsight profiling to find what actually bounds this network there.
