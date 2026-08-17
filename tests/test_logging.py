import json

from segdeploy.logging_utils import MetricsLogger


def test_log_and_read_roundtrip(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log("train_step", step=1, loss=0.5, lr=1e-4)
    logger.log("val_epoch", epoch=0, miou=0.71, iou_flat=0.9)

    records = MetricsLogger.read(tmp_path / "metrics.jsonl")
    assert len(records) == 2
    assert records[0]["kind"] == "train_step" and records[0]["loss"] == 0.5
    assert records[1]["miou"] == 0.71 and "wall_s" in records[1]


def test_append_survives_reopen(tmp_path):
    MetricsLogger(tmp_path).log("train_step", step=1, loss=1.0)
    MetricsLogger(tmp_path).log("train_step", step=2, loss=0.9)  # new logger, same file
    records = MetricsLogger.read(tmp_path / "metrics.jsonl")
    assert [r["step"] for r in records] == [1, 2]


def test_lines_are_valid_json(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log("config", lr="0.0003", size_hw="[512, 1024]")
    with open(tmp_path / "metrics.jsonl") as f:
        for line in f:
            json.loads(line)
