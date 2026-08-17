"""Plot training curves from a run's metrics.jsonl.

    pip install matplotlib
    python scripts/plot_training.py --run runs/fp32 --out docs/curves_fp32.png

Produces a two-panel figure: smoothed training loss + LR schedule (left),
val mIoU per epoch with per-class IoU traces (right). Multiple --run flags
overlay runs (e.g. FP32 baseline vs QAT fine-tune).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from segdeploy.labels import CATEGORY_NAMES
from segdeploy.logging_utils import MetricsLogger


def smooth(xs: list[float], beta: float = 0.98) -> list[float]:
    out, m = [], None
    for x in xs:
        m = x if m is None else beta * m + (1 - beta) * x
        out.append(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="Run dir with metrics.jsonl")
    ap.add_argument("--out", default="training_curves.png")
    ap.add_argument("--per-class", action="store_true", help="Overlay per-class IoU traces")
    args = ap.parse_args()

    fig, (ax_loss, ax_miou) = plt.subplots(1, 2, figsize=(12, 4.5))

    for run in args.run:
        records = MetricsLogger.read(Path(run) / "metrics.jsonl")
        steps = [r for r in records if r["kind"] == "train_step"]
        vals = [r for r in records if r["kind"] == "val_epoch"]
        label = Path(run).name

        if steps:
            xs = [r["step"] for r in steps]
            ax_loss.plot(xs, [r["loss"] for r in steps], alpha=0.25, lw=0.8)
            ax_loss.plot(xs, smooth([r["loss"] for r in steps]), lw=1.8, label=f"{label} loss")
        if vals:
            ex = [r["epoch"] + 1 for r in vals]
            ax_miou.plot(ex, [r["miou"] for r in vals], marker="o", lw=1.8, label=f"{label} mIoU")
            if args.per_class:
                for name in CATEGORY_NAMES:
                    ax_miou.plot(
                        ex, [r.get(f"iou_{name}") for r in vals], lw=0.9, alpha=0.5, label=name
                    )

    ax_loss.set(xlabel="iteration", ylabel="training loss")
    ax_miou.set(xlabel="epoch", ylabel="val IoU", ylim=(0, 1))
    for ax in (ax_loss, ax_miou):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
