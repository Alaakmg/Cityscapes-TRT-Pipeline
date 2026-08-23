"""Evaluate any backend (torch / onnx / trt) on the Cityscapes val split.

Always runs the deployed artifact itself, not the original PyTorch model.

    python -m segdeploy.evaluate --backend torch --checkpoint runs/fp32/best.pt --data-root ...
    python -m segdeploy.evaluate --backend onnx  --model model.onnx            --data-root ...
    python -m segdeploy.evaluate --backend trt   --model model_int8.engine     --data-root ...
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import CityscapesCategories
from .labels import CATEGORY_NAMES, NUM_CLASSES
from .metrics import ConfusionMatrix


def build_runner(args):
    if args.backend == "torch":
        import torch

        from .model import build_model_from_checkpoint
        from .runners import TorchRunner

        state = torch.load(args.checkpoint, map_location="cpu")
        model = build_model_from_checkpoint(state)
        model.load_state_dict(state.get("model", state))
        return TorchRunner(model)
    if args.backend == "onnx":
        from .runners import OnnxRunner

        return OnnxRunner(args.model)
    from .runners import TrtRunner

    return TrtRunner(args.model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "onnx", "trt"], required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--model")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only the first N images")
    ap.add_argument("--out", help="Write results (incl. confusion matrix) to this JSON file")
    args = ap.parse_args()

    runner = build_runner(args)
    ds = CityscapesCategories(args.data_root, "val", (args.height, args.width))
    dl = DataLoader(ds, batch_size=1, num_workers=2)

    cm = ConfusionMatrix(NUM_CLASSES)
    for i, (x, y) in enumerate(tqdm(dl)):
        if args.limit and i >= args.limit:
            break
        out = runner(x.numpy().astype(np.float32))
        # logits (N, C, H, W) -> argmax here; argmax-head engines return (N, H, W)
        pred = out.argmax(1) if out.ndim == 4 else out
        cm.update(y.numpy(), pred)

    print(cm.summary(CATEGORY_NAMES))

    if args.out:
        results = {
            "backend": args.backend,
            "model": args.model or args.checkpoint,
            "data_root": args.data_root,
            "size_hw": [args.height, args.width],
            "limit": args.limit,
            "miou": cm.miou(),
            "pixel_accuracy": cm.pixel_accuracy(),
            "iou_per_class": {
                n: float(v) for n, v in zip(CATEGORY_NAMES, cm.iou_per_class())
            },
            # rows = target, cols = prediction; kept so heatmaps and derived
            # metrics don't require re-running inference
            "confusion_matrix": cm.mat.tolist(),
            "class_names": CATEGORY_NAMES,
        }
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
