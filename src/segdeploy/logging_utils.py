"""Append-only JSONL metrics logging.

One JSON object per line in <out_dir>/metrics.jsonl. JSONL because it is
append-safe (a crashed run keeps everything logged so far) and loads with
pd.read_json(..., lines=True). scripts/plot_training.py consumes it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class MetricsLogger:
    def __init__(self, out_dir: str | Path, filename: str = "metrics.jsonl"):
        self.path = Path(out_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def log(self, kind: str, **fields) -> None:
        """Append one record. `kind` is e.g. 'train_step', 'val_epoch'."""
        record = {"kind": kind, "wall_s": round(time.time() - self._t0, 2), **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def read(path: str | Path) -> list[dict]:
        """Load all records from a metrics.jsonl file."""
        out = []
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
