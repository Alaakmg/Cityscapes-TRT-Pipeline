import pytest
import torch

from segdeploy.model import build_model


@pytest.fixture(scope="module")
def model():
    return build_model(num_classes=8, pretrained=False).eval()


def test_output_shape(model):
    x = torch.randn(2, 3, 64, 128)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (2, 8, 64, 128)


def test_output_shape_other_resolution(model):
    x = torch.randn(1, 3, 96, 160)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (1, 8, 96, 160)


def test_tiny_overfit():
    """Training on a trivially learnable target must drive loss to ~0.

    A constant-class target is learnable through the head bias alone, so this
    failing genuinely means the model/loss/optimizer wiring is broken rather
    than the test being unlucky.
    """
    torch.manual_seed(0)
    model = build_model(pretrained=False).train()
    x = torch.randn(2, 3, 64, 64)
    y = torch.full((2, 64, 64), 3, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = torch.nn.CrossEntropyLoss()

    first = crit(model(x), y).item()
    for _ in range(30):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < min(0.1, first * 0.2), (
        f"loss did not collapse: {first:.3f} -> {loss.item():.3f}"
    )


def test_fold_batchnorm_is_exact():
    """Folding BN into convs must leave the (eval-mode) function unchanged."""
    import numpy as np

    from segdeploy.model import fold_batchnorm

    torch.manual_seed(0)
    model = build_model(pretrained=False).eval()
    # give BN non-trivial statistics so folding actually has something to do
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-1, 1); m.running_var.uniform_(0.5, 2)
            m.weight.data.uniform_(0.5, 1.5); m.bias.data.uniform_(-0.5, 0.5)
    x = torch.randn(1, 3, 64, 128)
    with torch.inference_mode():
        ref = model(x)
    n = fold_batchnorm(model)
    with torch.inference_mode():
        out = model(x)
    assert n > 60
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=1e-4, atol=1e-4)
