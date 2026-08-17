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
