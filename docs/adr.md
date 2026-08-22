# Architecture decision records

One entry per decision that shaped this project, in the order I made them.
Each records the situation at the time, what I chose, what I didn't, and what
it cost or bought later. Status is `accepted` unless something overrode it.

Format: context, decision, alternatives, consequences.

---

## ADR-001: One deep deployment project instead of several training projects

**Date:** 2026-08 (project start) · **Status:** accepted

**Context.** My two earlier Cityscapes segmentation projects, both in
TensorFlow/Keras, were training projects: they end at a trained model and a
demo. Export, quantization, benchmarking on the target and profiling are a
different discipline, and they deserve a project of their own, measured end
to end, rather than an appendix at the bottom of a training repo.

**Decision.** Build a single end-to-end pipeline, PyTorch -> ONNX -> TensorRT
-> Jetson, on the same task I already know, and measure the quality/efficiency
trade-off at every step. Depth over breadth.

**Alternatives.** Several smaller projects (distillation, NAS, radar, mixed
precision) in parallel. Rejected: the marginal signal of a third half-finished
repo is low, and the deployment loop is where the interesting decisions live.

**Consequences.** Everything below inherits the constraint that the *measured
trade-off* is the product, not the model. Distillation and the radar
proof-of-concept became follow-ups that reuse this harness.

---

## ADR-002: Reimplement in PyTorch, keep the 8-category Cityscapes task

**Date:** 2026-08-17 · **Status:** accepted

**Context.** My prior models were Keras; the tooling this project leans on
(ONNX export, ModelOpt, the TensorRT samples) is PyTorch-first. Cityscapes
with the 8 top-level categories is the task I have baselines for (0.786 mIoU,
60.5M-param U-Net).

**Decision.** ResNet50-encoder U-Net in PyTorch, ImageNet-pretrained encoder,
AdamW + cosine, AMP. Same label mapping as before, single source of truth in
`labels.py`. Don't chase the old number, produce a clean reproducible
reference.

**Alternatives.** Port the Keras weights (no: I want a PyTorch training story,
not a conversion story). A lighter real-time architecture like DDRNet or
BiSeNet (deferred: continuity with my own baseline mattered more, and the
model's size makes the quantization effects visible).

**Consequences.** 44M params, val mIoU 0.8334 in 60 epochs, +4.7 points over
the old model with fewer parameters. The ResNet encoder later turned out to
carry the activation outliers that broke naive INT8 calibration (ADR-012) and
the residual structure that slows the QAT engine (ADR-016), both useful
findings I'd have missed with a simpler backbone.

---

## ADR-003: Export-friendly architecture constraints

**Date:** 2026-08-17 · **Status:** accepted

**Context.** Time spent fighting ONNX opsets is time not spent on
optimization.

**Decision.** Plain Conv/BN/ReLU, nearest-neighbour upsampling with an
*integer* `scale_factor` (never a dynamic `size=`), concat skips, one output
head, no deep supervision. No deformable convs, no custom ops.

**Alternatives.** Bilinear upsampling (exportable but generates Resize nodes
with coordinate-transform attributes that have historically varied across
opsets), deep supervision (multiple outputs complicate engine I/O and
calibration).

**Consequences.** Export was a non-event: max abs logit diff 4.1e-05, argmax
mismatch 3.8e-06. The profile later showed the decoder's 3x3 convs dominate
latency (66% on the Jetson), which is the price of a plain U-Net decoder at
full resolution, and the next thing to attack.

---

## ADR-004: Static input shape, 1x3x512x1024

**Date:** 2026-08-17 · **Status:** accepted

**Context.** Cameras run at a fixed resolution. Dynamic shapes mean TensorRT
optimization profiles, per-shape calibration, and more ways for INT8 to go
quietly wrong.

**Decision.** One static shape through the whole pipeline. One engine per
resolution if that ever changes.

**Consequences.** Simpler everything. The cost (one engine per resolution)
hasn't been paid yet.

---

## ADR-005: Fix the benchmark methodology before any experiment

**Date:** 2026-08-17 · **Status:** accepted

**Context.** Benchmark tables are easy to fudge by accident: measuring the
PyTorch model instead of the engine, mixing GPUs, reporting means without
tails, losing raw data.

**Decision.**
- Accuracy is always measured on the deployed artifact (ONNX session, TRT
  engine) through one backend-agnostic runner interface, never on the source
  model.
- Latency: batch 1 (the camera-feed setting), 20 warm-up + 200 timed
  iterations, explicit device sync, mean and p95.
- All desktop rows come from the same GPU type; all Jetson rows from the same
  power mode and clock state, stated in the table.
- Every stage writes JSON (per-class IoU, full confusion matrix, raw
  per-iteration latencies, engine build metadata). Nothing lives only in
  stdout. The empty table was in the README before the first training run.

**Alternatives.** Report trtexec numbers only (rejected for accuracy: trtexec
doesn't run the val set; kept as a cross-check).

**Consequences.** Every number in the README has a file behind it, and the
Jetson session could add power and profiling columns without changing a line
of the harness.

---

## ADR-006: One preprocessing function, shared by everything

**Date:** 2026-08-17 · **Status:** accepted

**Context.** Calibration data that doesn't match inference preprocessing is
the classic silent INT8 killer.

**Decision.** `segdeploy.data.preprocess_image` is the only resize+normalize
path: training, the parity check, and the INT8 calibrator all call it.

**Consequences.** The desktop calibration cache transferred to the Jetson and
produced matching accuracy (0.8236 vs 0.8246) with no calibration images on
the device.

---

## ADR-007: Rent GPUs per session; the 5090 over the cheaper card

**Date:** 2026-08-17 · **Status:** accepted

**Context.** My Mac has no CUDA. RunPod offered an RTX PRO 4500 at $0.72/h
(high availability) and an RTX 5090 at $0.99/h (low availability). I first
leaned toward the 4500 on availability.

**Decision.** 5090. Per-run cost is what matters, not per-hour: with roughly
half the memory bandwidth and compute, the 4500 would take ~2x longer and cost
*more* per run. Secure Cloud on-demand (not spot) so a run can't be preempted;
a 50 GB network volume in one datacenter for the dataset, checkpoints and
results; terminate the pod after every session since everything of value is
on the volume.

**Alternatives.** Spot instances (cheaper, but a 4-hour run interrupted at
hour 3 costs more than the discount saved). Keeping a pod stopped between
sessions (stopped pods lose their GPU reservation anyway).

**Consequences.** Full training run: 65 min, ~$1.10. Whole desktop table
including two QAT debugging sessions: under $5. One lesson: a stalled job
idling a pod costs more than the job (ADR-012, consequence 2).

---

## ADR-008: The Jetson is a measurement instrument, not a workstation

**Date:** 2026-08-18 · **Status:** accepted

**Context.** Orin Nano Super dev kit, 8 GB, microSD storage. NVMe prices in
2026 made an SSD a ~300 EUR question.

**Decision.** No SSD. Train and export on the rented GPU, move ONNX files to
the Jetson, build engines and measure there. The SD card only ever serves
boot, a one-time copy of the val split, and engines. The bottlenecks that
matter, 8 GB of RAM and a 68 GB/s memory bus, are exactly the ones to
measure against.

**Alternatives.** SSD for comfort (revisit only if container pulls start
costing iteration time). Training on the Jetson (no: wrong tool, and
production edge targets don't train either).

**Consequences.** Jetson sessions are short and cheap. Transfer of the 1.1 GB
val split over WiFi was a one-off.

---

## ADR-009: Engines are built on the device that runs them; the repo ships ONNX

**Date:** 2026-08-17 · **Status:** accepted

**Context.** TensorRT engines are tied to GPU architecture and TensorRT
version. An engine built on a desktop GPU will not run on a Jetson.

**Decision.** Build on target, every session, from the committed ONNX. Engines
are disposable; their `.meta.json` sidecars (precision, size, build time, TRT
version) are the provenance that gets committed.

**Consequences.** Engine builds turned out to take seconds after the first
build per machine (TensorRT's kernel-timing cache), so "rebuild everything
from ONNX" is free. That made it safe to terminate pods aggressively.

---

## ADR-010: Pin desktop TensorRT to 10.x to match JetPack

**Date:** 2026-08-17 · **Status:** accepted

**Context.** `pip install tensorrt` gave 11.2, which has removed
`IInt8EntropyCalibrator2` (implicit INT8 quantization is gone; explicit Q/DQ
only). JetPack ships TensorRT 10.x.

**Decision.** Pin desktop TRT to 10.x so both sides of the table share a major
version, and keep the entropy-calibration PTQ path reproducible. Treat
calibrator-based PTQ as legacy: the QAT/Q-DQ path is the one with a future.
Import the calibrator lazily so FP32/FP16 builds don't depend on it.

**Consequences.** Desktop 10.16.1 vs Jetson 10.16.2 in the end: same minor
version, one less variable. Also a documented finding about where TensorRT is
heading.

---

## ADR-011: Drop pycuda; torch tensors as TensorRT device buffers

**Date:** 2026-08-17 · **Status:** accepted

**Context.** pycuda has no wheels and needs nvcc + boost headers at install
time. The RunPod PyTorch image ships neither. Torch is already a hard
dependency everywhere the engines run.

**Decision.** Allocate I/O buffers with `torch.empty(..., device="cuda")`,
hand `data_ptr()` to TensorRT, execute on the current torch stream. Same
pattern in the INT8 calibrator.

**Alternatives.** cuda-python bindings (another dependency), fixing pycuda's
build (time).

**Consequences.** One less install step on every target, Jetson included,
where torch's standard cu130 wheel runs on Orin through PTX JIT (fine as an
allocator; I'd use NVIDIA's Jetson builds for torch *compute*). The pageable
numpy copies this pattern implies cost ~2.8 ms per frame on the Jetson -
measured, and on the list to fix with pinned buffers.

---

## ADR-012: QAT via ModelOpt, with calibration driven by hand

**Date:** 2026-08-18 · **Status:** accepted

**Context.** `pytorch-quantization` is deprecated and broken on current torch;
NVIDIA's successor is ModelOpt. Its default `INT8_DEFAULT_CFG` collapsed my
model from 0.833 to 0.18 mIoU before any training: per-tensor max calibration
set activation scales from ~1.3e3 BN outliers in ResNet `layer2`.

**Decision.**
1. Histogram calibration with percentile-99.9 clipping for activations, max
   for weights (same idea as TensorRT's entropy calibrator, which is why PTQ
   never had the problem). Restores 0.819 with zero training.
2. Drive calibration manually (`enable_stats_collection` + per-quantizer
   `load_calib_amax`) because ModelOpt's own calibration driver crashes on
   histogram calibrators.
3. Log a `calib_eval` record *before* fine-tuning, so a broken calibration is
   visible in minute two, not after five wasted epochs.
4. Export with a normal `torch.onnx.export` (Q/DQ nodes), build `--int8` with
   no calibrator.

**Alternatives.** Stay on pytorch-quantization (dead). Quantize the ONNX
directly with ModelOpt's ONNX tools (possible, but then QAT fine-tuning isn't
available).

**Consequences.** QAT engine 0.8290 vs PTQ 0.8246, and the fake-quant model
predicted its deployed accuracy within a tenth of a point. Two costly lessons
recorded: a broad `except ImportError` hid the real failure (an undeclared
`huggingface_hub` import) for 40 minutes of idle pod time, and ModelOpt config
patterns that don't match silently no-op, so assert your config applied.

---

## ADR-013: JetPack 7.2.1 over 6.2.1 on the Jetson

**Date:** 2026-08-20/21 · **Status:** accepted

**Context.** JetPack 6.2.1 is the mature Orin release (TensorRT 10.3, SD-card
image, documented headless first boot over USB-C). JetPack 7.2.1 was nine
days old on Orin: TensorRT 10.16.2, the same minor version as my desktop pin,
but installed via a new USB-stick ISO installer that also updates the QSPI
firmware.

**Decision.** JetPack 7.2.1, for the TensorRT version match. Accept the
newer-platform risk.

**Alternatives.** 6.2.1 first, 7 later (I had the image downloaded; it was the
fallback if the installer failed a third time).

**Consequences.** The risk materialized: the installer's firmware step
silently skipped on the first pass, leaving a 39.x OS on 36.4.7 firmware: text
console fine, GPU driver dead, no setup wizard. Root cause isolated by
changing one variable at a time (the firmware version on the UEFI screen).
Fix: wipe the card, reinstall, watch both capsule passes complete. Net cost:
one evening; net gain: identical TensorRT minor versions on both rows of the
table, and a reproducible install recipe. Side decisions made along the way:
a DisplayPort-to-USB-C cable into a USB-C portable monitor cannot drive UEFI
output (no source-side handshake), so native DP it is, and a USB-TTL serial
cable on the 12-pin header is the permanent fix for headless boot debugging.

---

## ADR-014: Report Jetson numbers at MAXN_SUPER with locked clocks; sweep the rest

**Date:** 2026-08-21/22 · **Status:** accepted

**Context.** The Orin Nano Super has three power modes (15 W, 25 W,
MAXN_SUPER) and a DVFS governor. Unstated power state makes Jetson benchmarks
incomparable.

**Decision.** The headline rows use MAXN_SUPER with `jetson_clocks` (GPU
1.02 GHz, EMC 3.2 GHz), stated in the table. Power is module input (`VDD_IN`)
from `tegrastats` at 2 Hz during the benchmark, reported as watts and mJ per
frame. Then sweep all modes, locked and DVFS, so the headline is a point on a
curve rather than a cherry-pick.

**Consequences.** The sweep produced the cleanest result in the project:
energy per frame is set by precision, not mode (INT8-PTQ ~260-280 mJ, FP16
~600-650 at every mode), the INT8 advantage is constant at 1.8x / 2.3x, and
Super mode buys latency, not efficiency. Locking clocks mostly buys p95
determinism.

---

## ADR-015: Decide INT8's fate on the target, not the desktop

**Date:** 2026-08-17 -> 2026-08-21 · **Status:** accepted

**Context.** On the RTX 5090, INT8-PTQ was 2.94 ms vs FP16's 2.96 ms.
Nothing. The easy conclusion was "INT8 isn't worth it for this model."

**Decision.** Don't conclude. Keep the INT8 rows, state the desktop result
honestly, and let the bandwidth-starved target decide.

**Consequences.** On the Jetson, INT8-PTQ is 1.8x faster than FP16 and 2.4x
cheaper per frame in energy; the same model now argues the other way. It
is the only configuration above 30 fps at 25 W. This is the project's
clearest argument for measuring on the deployment target.

---

## ADR-016: Diagnose the QAT engine's slowness instead of dropping the row

**Date:** 2026-08-21 · **Status:** accepted, resolved 2026-08-22 (v2 residual Q/DQ, v3 BN folding, v4 concat quantizers: 31.6 -> 19.9 ms on the Jetson, faster than PTQ at +0.6 mIoU)

**Context.** The QAT engine was slower than PTQ on both devices despite
identical size and, on the Jetson, identical per-conv timings.

**Decision.** Profile per layer (`trtexec --dumpProfile`) before forming an
opinion. Result: the encoder runs as 100 layers in the QAT engine vs 58 in
PTQ's (13.8 vs 6.5 ms). ModelOpt places Q/DQ on conv inputs and weights but
not on the residual-add inputs, so TensorRT can't fuse conv + add + ReLU in
the ResNet bottlenecks. Keep the row, document the cause, schedule the fix
(quantize the residual and concat paths).

**Consequences.** The table shows an honest weak spot with a known remedy,
which is worth more than a clean table. Expected outcome of the fix: PTQ's
latency with QAT's accuracy, the best point on the chart.

---

## ADR-017: The commit log is part of the deliverable

**Date:** 2026-08-17 · **Status:** accepted

**Context.** A deployment project is judged on its debugging trail as much
as on its final table, and the repo should show both.

**Decision.** Tests for the parts that can run on a CPU (label mapping,
metrics math, model shapes, ONNX export + parity) in CI on every push; lint
enforced. Raw result JSONs and profiles committed; datasets and engines never.
Every first-contact fix (pycuda, TRT 11, `IHostMemory.len`, ModelOpt) is its
own commit with a message explaining what broke. Findings written up as they
happen, with the numbers traceable to files.

**Consequences.** The commit log reads as a debugging trail across the
deployment stack. This document exists because that trail deserved a
narrative.

---

## ADR-018: Pinned host buffers in the runner; keep the argmax export, don't sell it

**Date:** 2026-08-22 · **Status:** accepted

**Context.** The Jetson profile showed ~2.8 ms/frame of harness overhead over
trtexec and a 16.8 MB logits tensor copied back every frame.

**Decision.** Host buffers are pinned torch tensors, copies run async on a
private stream. Measured: -2.5 ms on every engine, within 0.6 ms of trtexec;
INT8-PTQ 20.4 ms, INT8-QAT crosses 30 fps. The on-device argmax export stays
(a consumer wants the mask, and it matters over PCIe) but is documented as a
wash at batch 1 on unified memory: the D2H saving is eaten by the reduction.

**Consequences.** A wrapped-module export silently invalidated the INT8
calibration cache (renamed tensors) and produced an FP16 engine labelled int8.
Resolved by graph surgery that preserves names, a test that asserts it, and a
per-precision layer histogram persisted with every engine, which also
quantified the QAT problem at 48 FP16 layers. The desktop rows still use the
v1 harness; they get re-measured on the next pod session so the table has one
methodology.

---

## Open decisions

- **Decoder width:** with QAT resolved, the profile's remaining target is
  the decoder (66% of time). ModelOpt lesson folded into ADR-012's list:
  any pre-existing quantizer makes quantize/restore a no-op.
- **Decoder width:** the profile says the decoder, not the backbone, is
  what to shrink. A thinner dec0/dec1 is the next architecture experiment, with Jetson
  latency as the objective.
