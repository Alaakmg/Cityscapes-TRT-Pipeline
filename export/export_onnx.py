"""Export the trained model to ONNX.

Static input shape on purpose: inference runs at a fixed camera resolution,
and static shapes keep TensorRT profiles and INT8 calibration simple.

    python export/export_onnx.py --checkpoint runs/fp32/best.pt --out model.onnx
"""

from __future__ import annotations

import argparse

import onnx
import torch

from segdeploy.model import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = build_model(pretrained=False).eval()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state))

    dummy = torch.randn(1, 3, args.height, args.width)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        opset_version=args.opset,
        input_names=["image"],
        output_names=["logits"],
        do_constant_folding=True,
        dynamo=False,
    )

    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    print(f"Exported and checked: {args.out}")
    print(f"  inputs : {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]}")
    print(f"  outputs: {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]}")


if __name__ == "__main__":
    main()
