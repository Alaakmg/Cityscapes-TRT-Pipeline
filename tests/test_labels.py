import numpy as np
import pytest

from segdeploy.labels import NUM_CLASSES, colorize, encode_labels


def test_known_mappings():
    raw = np.array([[7, 11, 23], [26, 24, 0]])  # road, building, sky / car, person, unlabeled
    out = encode_labels(raw)
    assert out.tolist() == [[1, 2, 5], [7, 6, 0]]


def test_full_range_maps_into_categories():
    raw = np.arange(-1, 34).reshape(1, -1)
    out = encode_labels(raw)
    assert out.min() >= 0 and out.max() < NUM_CLASSES
    assert out[0, 0] == 7  # -1 license plate -> vehicle


def test_invalid_id_raises():
    with pytest.raises(ValueError):
        encode_labels(np.array([[99]]))


def test_colorize_shape():
    mask = np.zeros((4, 6), dtype=np.uint8)
    assert colorize(mask).shape == (4, 6, 3)
