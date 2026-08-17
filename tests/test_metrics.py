import numpy as np

from segdeploy.metrics import ConfusionMatrix


def test_perfect_prediction():
    cm = ConfusionMatrix(3)
    y = np.array([0, 1, 2, 2, 1])
    cm.update(y, y)
    assert cm.miou() == 1.0
    assert cm.pixel_accuracy() == 1.0


def test_known_iou():
    # class 0: tp=1, fn=1, fp=0 -> IoU 0.5 ; class 1: tp=1, fn=0, fp=1 -> IoU 0.5
    cm = ConfusionMatrix(2)
    cm.update(np.array([0, 0, 1]), np.array([0, 1, 1]))
    np.testing.assert_allclose(cm.iou_per_class(), [0.5, 0.5])


def test_streaming_equals_single_shot():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 4, size=1000)
    p = rng.integers(0, 4, size=1000)
    a = ConfusionMatrix(4); a.update(t, p)
    b = ConfusionMatrix(4)
    b.update(t[:300], p[:300]); b.update(t[300:], p[300:])
    assert np.array_equal(a.mat, b.mat)


def test_confusion_matrix_serializes():
    """cm.mat.tolist() is what evaluate.py --out persists; it must round-trip JSON."""
    import json

    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 1, 2]), np.array([0, 1, 1]))
    restored = np.array(json.loads(json.dumps(cm.mat.tolist())))
    assert np.array_equal(restored, cm.mat)
