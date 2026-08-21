"""End-to-end export + parity test on a small input (CPU, no dataset needed)."""
import numpy as np
import pytest
import torch

from segdeploy.model import build_model

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def test_export_and_parity(tmp_path):
    model = build_model(pretrained=False).eval()
    h, w = 64, 128
    path = tmp_path / "m.onnx"
    torch.onnx.export(
        model, torch.randn(1, 3, h, w), str(path),
        opset_version=17, input_names=["image"], output_names=["logits"], dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal((1, 3, h, w)).astype(np.float32)
    with torch.inference_mode():
        ref = model(torch.from_numpy(x)).numpy()
    out = sess.run(None, {"image": x})[0]

    assert np.abs(ref - out).max() < 1e-3
    assert (ref.argmax(1) != out.argmax(1)).mean() < 1e-3


def test_argmax_head_export(tmp_path):
    """The --argmax export must produce an int32 (N, H, W) mask equal to argmax of the logits."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
    from export_onnx import ArgmaxHead

    model = build_model(pretrained=False).eval()
    h, w = 64, 128
    path = tmp_path / "m_argmax.onnx"
    torch.onnx.export(
        ArgmaxHead(model).eval(), torch.randn(1, 3, h, w), str(path),
        opset_version=17, input_names=["image"], output_names=["mask"], dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(1).standard_normal((1, 3, h, w)).astype(np.float32)
    mask = sess.run(None, {"image": x})[0]
    with torch.inference_mode():
        ref = model(torch.from_numpy(x)).argmax(1).numpy()

    assert mask.shape == (1, h, w) and mask.dtype == np.int32
    assert (mask != ref).mean() < 1e-3
