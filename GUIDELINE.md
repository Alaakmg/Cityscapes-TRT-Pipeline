# Step-by-Step Guideline — From Repo to Application

Target role: ML Engineer, Model Optimization (Zendar, Paris)
Estimated total: 6–9 weeks part-time · GPU budget ≈ €30 · Hardware budget ≈ €250 (Jetson)

Legend: ☐ do · ✅ checkpoint (don't advance until true)

---

## Phase 0 — Setup (Days 1–3, do everything in parallel)

1. ☐ Register on cityscapes-dataset.com (approval takes 1–3 days — do this FIRST).
2. ☐ Create a GitHub repo, unzip `cityscapes-trt-pipeline.zip`, initial commit, push.
3. ☐ Confirm the CI badge goes green (the workflow runs ruff + the 11 unit tests).
4. ☐ On the M2 Max: `pip install -e ".[dev]" && pytest -q` — your local dev loop.
5. ☐ Create a RunPod account; create a **50 GB network volume** first (it pins your
   region), note the region.
6. ☐ Order the Jetson Orin Nano dev kit (~€250) so shipping overlaps Phases 1–3.
7. ☐ Read the repo end to end. You must be able to explain every file — this code
   is your interview script.

✅ Checkpoint: CI green on GitHub, Cityscapes approval email received, Jetson ordered.

---

## Phase 1 — FP32 baseline on RunPod (Week 1–2)

8. ☐ Download Cityscapes (`leftImg8bit_trainvaltest` + `gtFine_trainvaltest`,
   ~11 GB) onto the network volume, keep the official layout.
9. ☐ Rent an **RTX 5090, Secure Cloud, on-demand** (fallback: A40 at $0.44/hr),
   attach the volume, `git clone` your repo, `pip install -e ".[dev]"`.
10. ☐ Point `configs/baseline.yaml` `data_root` at the volume mount and `out_dir`
    at a volume path (checkpoints must survive the pod).
11. ☐ **Smoke test: run 1 epoch** (`epochs: 1`). Verify loss decreases, val mIoU
    prints per class, `last.pt` lands on the volume, GPU utilization is high
    (`nvidia-smi` — if <80%, raise `workers`).
12. ☐ Launch the real run (60 epochs, ~3–5 h, ~$4–6). Use tmux so SSH drops
    don't kill it.
13. ☐ Record in the README table: val mIoU, per-class IoU, params, checkpoint size,
    and PyTorch GPU latency (`python -m segdeploy.benchmark --backend torch ...`).
14. ☐ Commit the filled first table row + a `docs/findings.md` note on anything
    surprising (which classes are weakest — expect object/human).

✅ Checkpoint: val mIoU ≥ ~0.75 (sanity vs your old 0.767/0.786 TF runs) and a
reproducible `best.pt` on the volume. If mIoU is far below, debug the label
mapping before proceeding — nothing downstream is meaningful otherwise.

---

## Phase 2 — ONNX export + parity (Week 2, ~3 evenings, runs on the Mac)

15. ☐ `python export/export_onnx.py --checkpoint best.pt --out model.onnx`
16. ☐ `python export/check_parity.py ...` — must pass (tol 1e-3, zero argmax drift).
17. ☐ Evaluate the ONNX artifact itself on val
    (`python -m segdeploy.evaluate --backend onnx ...`) — mIoU must match the
    PyTorch row to ~3 decimals. Fill the ONNX Runtime table row.
18. ☐ Document any export friction (opset issues, warnings) in `docs/findings.md`.
19. ☐ Commit `model.onnx` artifacts to the volume + a GitHub Release (not git).

✅ Checkpoint: parity script exits 0; ONNX mIoU == PyTorch mIoU.

---

## Phase 3 — TensorRT FP32 / FP16 / INT8 on RunPod (Weeks 3–4, the heart)

Rent the SAME GPU type as Phase 1 for every TRT number (comparability).

20. ☐ Build FP32 engine (`trt/build_engine.py`). Verify mIoU matches ONNX and
    latency beats ONNX Runtime. Fill row.
21. ☐ Build FP16 engine (`--fp16`). Evaluate the ENGINE on val; fill row. Note
    per-class deltas — thin structures (object, human) degrade first.
22. ☐ **INT8 PTQ:** build with `--int8 --calib-dir .../leftImg8bit/train
    --calib-images 500`. Evaluate; fill row. Keep `calibration.cache` on the volume.
23. ☐ **INT8 QAT:** install `pytorch-quantization`, run `trt/qat_finetune.py`
    (5 epochs from `best.pt`, <$1), re-export ONNX (Q/DQ preserved), rebuild
    `--int8` (no calibrator needed), evaluate; fill row.
24. ☐ Write the quantization findings section: FP32→PTQ drop, QAT recovery,
    per-class breakdown. If time allows: leave the most sensitive layer in FP16
    and show the recovery (bridge to the mixed-precision follow-up).
25. ☐ Optional but cheap: eval the INT8 engine on BDD100K/ACDC val as a
    domain-shift robustness note.

✅ Checkpoint: 6 desktop rows filled; the FP32→PTQ→QAT story is visible in numbers.
Fallback: if QAT fights you, ship PTQ-only and mark QAT "in progress" — documented
partial results beat silence.

---

## Phase 4 — Jetson deployment (Week 5, needs the hardware)

26. ☐ Flash JetPack, verify `python3 -c "import tensorrt"` works on-device.
27. ☐ Copy `model.onnx` + calibration cache + a small val subset to the Jetson.
    **Rebuild engines ON the Jetson** (engines are not portable — say so in README).
28. ☐ Re-run the full benchmark + eval suite on-device (FP16 and INT8 rows).
    Batch 1. Record power via `tegrastats` if possible.
29. ☐ Profile: `trtexec --loadEngine=... --dumpProfile` and Nsight Systems.
    Identify the top-3 layers by time; state memory-bound vs compute-bound.
30. ☐ Make ONE profiling-driven change (e.g. resolution trade-off, preprocessing
    fusion) and record before/after. This single step is the strongest interview
    material in the whole project.

✅ Checkpoint: Jetson rows filled + a written bottleneck analysis with one
implemented improvement.

---

## Phase 5 — C++ wrapper + polish (Week 6)

31. ☐ Build `cpp/` on the Jetson (`cmake -B build && cmake --build build`),
    run it on a val image, commit the output mask image to `docs/`.
32. ☐ Final README: complete table at top, latency-vs-mIoU plot, limitations +
    next-steps section (mixed precision, distillation, DLA).
33. ☐ `git tag v1.0`. Restart-and-run any analysis notebook before committing.
34. ☐ Write one technical blog post (in English) walking through the findings —
    doubles as the communication-skills evidence the posting asks for.

✅ Checkpoint: a stranger can go from README to reproduced desktop numbers.

---

## Phase 6 — Second project: distillation (Weeks 7–8, reuses everything)

35. ☐ Add a student model (Fast-SCNN-scale, ~1–3M params) to `segdeploy`.
36. ☐ Train it twice: from labels only, then distilled from your 44M teacher
    (soft targets + feature hints). Same data, same harness.
37. ☐ Run both students through the SAME pipeline (ONNX → TRT INT8 → Jetson).
38. ☐ Deliverable: a Pareto plot (params/latency vs mIoU) with teacher, both
    students, and your U-Net variants — the "trade-off between model quality
    and computational cost" figure, literally the posting's language.
39. ☐ Optional memorability play (only if energy remains): CARRADA radar
    proof-of-concept — data loader, small baseline, honest README results.

---

## Phase 7 — Application (Week 8–9)

40. ☐ CV bullet: "Took a 44M-param segmentation model from FP32 PyTorch to INT8
    TensorRT on Jetson Orin Nano: X× speedup at Y-point mIoU cost" — every number
    backed by the repo.
41. ☐ Cover note in English, explicitly mapping: pipeline repo → their
    PyTorch/ONNX/TensorRT stack; QAT/sensitivity findings → their "numerical
    differences" bullet; distillation Pareto → their "quality vs cost" bullet;
    Jetson profiling → their "real hardware measurements" bullet.
42. ☐ Links: repo (CI badge visible), blog post, and your two original projects
    framed as "training foundations" that this work extends.
43. ☐ Interview prep: rehearse the one-sentence pipeline summary; be ready to
    defend every design decision in the README (static shapes, calibration
    preprocessing, engines-built-on-device, plain-conv architecture).

---

## Standing rules (apply throughout)

- Logic lives in `segdeploy/`; notebooks only visualize. Scripts stay headless.
- Every accuracy number comes from evaluating the deployed artifact, never the
  source PyTorch model.
- Same GPU type for all desktop TRT rows. Jetson rows are the story.
- Checkpoints, ONNX, calibration cache → network volume. Engines are disposable.
- Commit findings the day you get them; a stale `docs/findings.md` never recovers.
- If a week slips: cut Phase 6's radar option first, then the C++ wrapper —
  never cut the parity checks or the per-class quantization analysis.
