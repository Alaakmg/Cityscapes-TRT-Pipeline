"""Quantization-aware training with NVIDIA ModelOpt.

pytorch-quantization is deprecated and broken on current torch; modelopt is
its successor. mtq.quantize() inserts fake-quant modules for weights and
activations, a short calibration pass sets the initial ranges, fine-tuning
recovers the PTQ accuracy loss, and a normal torch.onnx.export writes
standard Q/DQ nodes. TensorRT consumes that as an explicit-quantization INT8
engine, no calibrator involved (ranges are in the graph).

    pip install nvidia-modelopt
    python trt/qat_finetune.py --config configs/runpod.yaml \
        --checkpoint /workspace/runs/fp32/best.pt --epochs 5 --out /workspace/runs/qat

Then:
    python trt/build_engine.py --onnx <out>/model_qat.onnx \
        --out model_int8_qat.engine --int8
"""

from __future__ import annotations

import argparse
import copy
import types
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.calib.histogram import HistogramCalibrator
    from modelopt.torch.quantization.model_calib import enable_stats_collection
    from modelopt.torch.quantization.nn import TensorQuantizer
except ImportError as e:  # pragma: no cover
    # keep the underlying error visible: modelopt itself may be installed while
    # one of its undeclared imports (e.g. huggingface_hub) is missing
    raise SystemExit(
        f"modelopt import failed: {e!r}\n"
        "for QAT: pip install nvidia-modelopt huggingface_hub"
    ) from e

from torchvision.models.resnet import Bottleneck

from segdeploy.data import CityscapesCategories
from segdeploy.labels import CATEGORY_NAMES
from segdeploy.logging_utils import MetricsLogger
from segdeploy.model import build_model
from segdeploy.train import evaluate, set_seed


def _bottleneck_forward_quant(self: Bottleneck, x: torch.Tensor) -> torch.Tensor:
    identity = x
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    if self.downsample is not None:
        identity = self.downsample(x)
    # Q/DQ on the residual branch: with both add inputs quantized TensorRT
    # fuses conv3 + add + relu into a single INT8 kernel. Without it the add
    # runs in FP16 as its own layer with re-quantization around it (measured:
    # 48 FP16 layers, encoder 2x slower than the PTQ engine).
    out = out + self.residual_input_quantizer(identity)
    return self.relu(out)


def add_residual_quantizers(model: nn.Module) -> int:
    """Give every ResNet bottleneck a quantizer on its identity branch.

    Must run *before* mtq.quantize / mto.restore. The attribute name ends in
    `input_quantizer` on purpose: modelopt's quant_cfg rules match quantizers
    by name, so `*input_quantizer` picks these up and gives them the same
    8-bit histogram config as every conv input, no extra config needed.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, Bottleneck):
            m.residual_input_quantizer = TensorQuantizer()
            m.forward = types.MethodType(_bottleneck_forward_quant, m)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="FP32 checkpoint to start from")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--calib-batches", type=int, default=64)
    ap.add_argument("--percentile", type=float, default=99.9, help="Activation amax percentile")
    ap.add_argument("--out", default="runs/qat")
    ap.add_argument(
        "--no-residual-quant", action="store_true",
        help="v1 behaviour: leave the residual adds unquantized (slower engine)",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size_hw = tuple(cfg["size_hw"])

    model = build_model(pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    if not args.no_residual_quant:
        print(f"residual quantizers added to {add_residual_quantizers(model)} bottlenecks")

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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(out_dir)
    logger.log(
        "config", start_checkpoint=args.checkpoint, epochs=args.epochs,
        lr=args.lr, calib_batches=args.calib_batches, percentile=args.percentile,
        quant="modelopt INT8, histogram/percentile activation calibration",
        residual_quant=not args.no_residual_quant,
    )

    # INT8_DEFAULT_CFG calibrates activations with MaxCalibrator, and this
    # model has ~1e3-magnitude activation outliers (torchvision-ResNet BN
    # channels in layer2): per-tensor scales set by those outliers collapse
    # val mIoU from 0.833 to 0.18. Histogram + percentile clipping restores
    # ~0.82 before fine-tuning even starts (measured, see docs/findings.md).
    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    for entry in cfg["quant_cfg"]:
        if entry.get("quantizer_name") == "*input_quantizer":
            entry["cfg"]["calibrator"] = "histogram"
    # modelopt's built-in max-calib driver crashes on histogram calibrators
    # (compute_amax() without a method), so calibration is driven manually.
    cfg["algorithm"] = None

    model = mtq.quantize(model, cfg, forward_loop=None)
    enable_stats_collection(model)
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(train_dl):
            if i >= args.calib_batches:
                break
            model(x.to(device))
    for m in model.modules():
        if not isinstance(m, TensorQuantizer) or getattr(m, "_disabled", False):
            continue
        cal = getattr(m, "_calibrator", None)
        if cal is None:
            continue
        if isinstance(cal, HistogramCalibrator):
            m.load_calib_amax("percentile", percentile=args.percentile)
        elif cal.compute_amax() is not None:
            m.load_calib_amax()
        m.enable_quant()
        m.disable_calib()

    cm = evaluate(model, val_dl, device)
    print(f"post-calibration (pre-QAT) val mIoU: {cm.miou():.4f}")
    logger.log("calib_eval", miou=round(cm.miou(), 5))

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Plain FP32 fine-tuning (no AMP: keeps the fake-quant numerics simple).
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
                logger.log(
                    "train_step", step=global_step, epoch=epoch,
                    loss=round(loss.item(), 5), lr=args.lr,
                )

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
            # mto.save keeps the modelopt state alongside the weights, so the
            # quantized model can be rebuilt exactly for export.
            mto.save(model, out_dir / "best_qat.pth")
            print(f"  -> new best ({best:.4f}), saved {out_dir / 'best_qat.pth'}")

    # Export the best checkpoint with Q/DQ nodes (CPU, same recipe as
    # export/export_onnx.py: static shape, opset 17, TorchScript exporter).
    h, w = size_hw
    fresh = build_model(pretrained=False)
    if not args.no_residual_quant:
        add_residual_quantizers(fresh)
    export_model = mto.restore(fresh, out_dir / "best_qat.pth").eval()
    onnx_path = out_dir / "model_qat.onnx"
    torch.onnx.export(
        export_model,
        torch.randn(1, 3, h, w),
        str(onnx_path),
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported {onnx_path} (best val mIoU {best:.4f})")


if __name__ == "__main__":
    main()
