"""Train the FP32 baseline.

    python -m segdeploy.train --config configs/baseline.yaml

Config-driven and seeded, AMP on CUDA, cosine LR after linear warmup,
keeps the best checkpoint by val mIoU.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import CityscapesCategories
from .labels import CATEGORY_NAMES, NUM_CLASSES
from .logging_utils import MetricsLogger
from .metrics import ConfusionMatrix
from .model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> ConfusionMatrix:
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES)
    for x, y in loader:
        logits = model(x.to(device))
        pred = logits.argmax(1).cpu().numpy()
        cm.update(y.numpy(), pred)
    return cm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size_hw = tuple(cfg["size_hw"])

    train_ds = CityscapesCategories(cfg["data_root"], "train", size_hw, augment=True)
    val_ds = CityscapesCategories(cfg["data_root"], "val", size_hw, augment=False)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], num_workers=cfg.get("workers", 4))

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    epochs = cfg["epochs"]
    warmup_iters = cfg.get("warmup_iters", 500)
    total_iters = epochs * len(train_dl)

    def lr_lambda(it: int) -> float:
        if it < warmup_iters:
            return it / max(warmup_iters, 1)
        p = (it - warmup_iters) / max(total_iters - warmup_iters, 1)
        return 0.5 * (1 + np.cos(np.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(out_dir)
    logger.log("config", **{k: str(v) for k, v in cfg.items()})
    best_miou = 0.0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"epoch {epoch + 1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1
            if global_step % 20 == 0:
                logger.log(
                    "train_step", step=global_step, epoch=epoch,
                    loss=round(loss.item(), 5), lr=sched.get_last_lr()[0],
                )
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        cm = evaluate(model, val_dl, device)
        miou = cm.miou()
        iou = cm.iou_per_class()
        logger.log(
            "val_epoch", epoch=epoch, step=global_step, miou=round(miou, 5),
            pixel_acc=round(cm.pixel_accuracy(), 5),
            **{f"iou_{n}": round(float(v), 5) for n, v in zip(CATEGORY_NAMES, iou)},
        )
        print(f"epoch {epoch + 1}: val mIoU={miou:.4f}")
        print(cm.summary(CATEGORY_NAMES))

        torch.save({"model": model.state_dict(), "epoch": epoch, "miou": miou}, out_dir / "last.pt")
        if miou > best_miou:
            best_miou = miou
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "miou": miou}, out_dir / "best.pt"
            )
            print(f"  -> new best ({best_miou:.4f}), saved {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
