"""Quantization-aware training.

Uses NVIDIA's pytorch-quantization toolkit: it swaps in quantized conv/linear
modules that carry Q/DQ nodes through ONNX export, which TensorRT consumes to
build an INT8 engine with learned ranges. This usually recovers most of the
accuracy lost to PTQ.

    pip install nvidia-pyindex && pip install pytorch-quantization
    python trt/qat_finetune.py --config configs/baseline.yaml \
        --checkpoint runs/fp32/best.pt --epochs 5 --out runs/qat

Then export with export/export_onnx.py (Q/DQ nodes are preserved) and build
with trt/build_engine.py --int8 (no calibrator needed: ranges are in the graph).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from pytorch_quantization import quant_modules
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "pytorch-quantization is required for QAT: "
        "pip install nvidia-pyindex && pip install pytorch-quantization"
    ) from e

from segdeploy.data import CityscapesCategories
from segdeploy.labels import CATEGORY_NAMES
from segdeploy.logging_utils import MetricsLogger
from segdeploy.train import evaluate, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="FP32 checkpoint to start from")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default="runs/qat")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Must be enabled BEFORE the model is constructed: replaces nn.Conv2d etc.
    # with quantized equivalents that fake-quantize weights and activations.
    quant_modules.initialize()
    from segdeploy.model import build_model  # import after initialize()

    model = build_model(pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)

    size_hw = tuple(cfg["size_hw"])
    train_dl = DataLoader(
        CityscapesCategories(cfg["data_root"], "train", size_hw, augment=True),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("workers", 4),
        drop_last=True,
    )
    val_dl = DataLoader(
        CityscapesCategories(cfg["data_root"], "val", size_hw),
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("workers", 4),
    )

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(out_dir)
    logger.log("config", start_checkpoint=args.checkpoint, epochs=args.epochs, lr=args.lr)

    best = 0.0
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for x, y in tqdm(train_dl, desc=f"QAT epoch {epoch + 1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            global_step += 1
            if global_step % 20 == 0:
                logger.log("train_step", step=global_step, epoch=epoch, loss=round(loss.item(), 5), lr=args.lr)

        cm = evaluate(model, val_dl, device)
        miou = cm.miou()
        iou = cm.iou_per_class()
        logger.log(
            "val_epoch", epoch=epoch, step=global_step, miou=round(miou, 5),
            pixel_acc=round(cm.pixel_accuracy(), 5),
            **{f"iou_{n}": round(float(v), 5) for n, v in zip(CATEGORY_NAMES, iou)},
        )
        print(f"QAT epoch {epoch + 1}: val mIoU={miou:.4f}")
        print(cm.summary(CATEGORY_NAMES))
        if miou > best:
            best = miou
            torch.save({"model": model.state_dict(), "miou": miou}, out_dir / "best.pt")


if __name__ == "__main__":
    main()
