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
