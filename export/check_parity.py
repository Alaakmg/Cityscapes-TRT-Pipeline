"""Check PyTorch <-> ONNX Runtime numerical parity.

Reports max-abs / max-rel logits difference and the fraction of pixels whose
argmax class changed. Exits non-zero if tolerances are exceeded, so it can
run in CI and as a pre-deployment gate.

    python export/check_parity.py --checkpoint runs/fp32/best.pt --onnx model.onnx
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from segdeploy.model import build_model
from segdeploy.runners import OnnxRunner, TorchRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--max-argmax-mismatch", type=float, default=1e-4)
    ap.add_argument("--out", help="Write parity stats to this JSON file")
    args = ap.parse_args()

    model = build_model(pretrained=False).eval()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    torch_runner = TorchRunner(model, device="cpu")
    onnx_runner = OnnxRunner(args.onnx)

    rng = np.random.default_rng(0)
    worst_abs, worst_rel, worst_mismatch = 0.0, 0.0, 0.0
    for _ in range(args.batches):
        x = rng.standard_normal((1, 3, args.height, args.width)).astype(np.float32)
        a, b = torch_runner(x), onnx_runner(x)
        diff = np.abs(a - b)
        worst_abs = max(worst_abs, float(diff.max()))
        worst_rel = max(worst_rel, float((diff / (np.abs(a) + 1e-6)).max()))
        worst_mismatch = max(worst_mismatch, float((a.argmax(1) != b.argmax(1)).mean()))

    print(f"max |logit diff|        : {worst_abs:.3e}  (tol {args.atol:.1e})")
    print(f"max relative diff       : {worst_rel:.3e}")
    print(f"argmax mismatch fraction: {worst_mismatch:.3e}  (tol {args.max_argmax_mismatch:.1e})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "onnx": args.onnx,
                    "checkpoint": args.checkpoint,
                    "batches": args.batches,
                    "max_abs_diff": worst_abs,
                    "max_rel_diff": worst_rel,
                    "argmax_mismatch_fraction": worst_mismatch,
                    "atol": args.atol,
                    "passed": bool(
                        worst_abs <= args.atol
                        and worst_mismatch <= args.max_argmax_mismatch
                    ),
                },
                f,
                indent=2,
            )
        print(f"wrote {args.out}")

    if worst_abs > args.atol or worst_mismatch > args.max_argmax_mismatch:
        print("PARITY CHECK FAILED", file=sys.stderr)
        sys.exit(1)
    print("Parity check passed.")


if __name__ == "__main__":
    main()
