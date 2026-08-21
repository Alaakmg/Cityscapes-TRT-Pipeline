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


def add_argmax_output(model: onnx.ModelProto) -> onnx.ModelProto:
    """Append ArgMax + Cast(int32) to an exported logits graph, in place.

    (N, C, H, W) logits -> (N, H, W) int32 mask. On a bandwidth-bound device
    the 16.8 MB FP32 logits tensor coming back every frame is waste when the
    consumer only wants the mask; int32 is 8x smaller.

    Done as graph surgery on the existing export rather than by wrapping the
    torch module: wrapping renames every tensor (/model/... prefix), which
    silently invalidates an INT8 calibration cache and TensorRT then builds an
    all-FP16 engine without complaint. Surgery keeps the names, so the cache
    built for the logits graph applies unchanged.
    """
    g = model.graph
    assert len(g.output) == 1, "expected a single logits output"
    logits = g.output[0]
    n, _c, h, w = [d.dim_value for d in logits.type.tensor_type.shape.dim]
    g.node.extend([
        onnx.helper.make_node("ArgMax", [logits.name], ["mask_i64"], axis=1, keepdims=0),
        onnx.helper.make_node("Cast", ["mask_i64"], ["mask"], to=onnx.TensorProto.INT32),
    ])
    del g.output[:]
    g.output.append(onnx.helper.make_tensor_value_info("mask", onnx.TensorProto.INT32, [n, h, w]))
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--argmax", action="store_true", help="Export the class mask instead of logits")
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
    if args.argmax:
        m = add_argmax_output(m)
        onnx.save(m, args.out)
    onnx.checker.check_model(m)
    print(f"Exported and checked: {args.out}")
    print(f"  inputs : {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]}")
    print(f"  outputs: {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]}")


if __name__ == "__main__":
    main()
