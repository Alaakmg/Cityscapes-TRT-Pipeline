"""Energy per frame from a tegrastats log captured around a benchmark.

The original sweep averaged VDD_IN over every sample in the log. The logs
start and end at idle (tegrastats is launched before the runner loads the
engine and killed after it exits), so that mean under-reports the power
drawn while the GPU is actually working, and the shorter the run the worse
it gets: an INT8 run of 5 s in a 10 s log is half idle at ~6 W.

Method here: energy above the pre-run idle baseline, integrated over the
whole log, is attributed to the benchmark's known duration
(warmup + timed iterations) x mean latency. Integrating the whole log makes
the number insensitive to the sensor's ramp at the start and decay at the
end of the run, which a "busy samples only" mean is not.

    P_idle = mean VDD_IN before the first GPU-busy sample
    E_dyn  = sum_i (P_i - P_idle) * dt          over the whole log
    T_run  = (warmup + iters) * latency_mean
    W      = P_idle + E_dyn / T_run
    mJ     = W * latency_mean

GPU-busy is GR3D_FREQ utilisation > 0. The busy-window mean is reported
alongside as a cross-check; it sits a few percent below W on short runs.

Usage:
    python -m segdeploy.power results/jetson_orin_nano/trt_int8_ptq ...
    python -m segdeploy.power --rebuild-sweep results/jetson_orin_nano/sweep
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_TS = "%m-%d-%Y %H:%M:%S"
_VDD = re.compile(r"VDD_IN (\d+)mW")
_GPU = re.compile(r"GR3D_FREQ (\d+)%")


@dataclass
class Sample:
    t: datetime
    watts: float
    gpu_pct: int


def parse_tegrastats(path: str | Path) -> list[Sample]:
    out = []
    for line in Path(path).read_text().splitlines():
        v, g = _VDD.search(line), _GPU.search(line)
        if not (v and g):
            continue
        t = datetime.strptime(line[:19], _TS).replace(tzinfo=timezone.utc)
        out.append(Sample(t, int(v.group(1)) / 1000.0, int(g.group(1))))
    if not out:
        raise ValueError(f"no VDD_IN samples in {path}")
    return out


def energy_from_log(
    log: str | Path, latency_ms_mean: float, iters: int = 200, warmup: int = 20
) -> dict:
    s = parse_tegrastats(log)
    n = len(s)
    span = (s[-1].t - s[0].t).total_seconds()
    dt = span / (n - 1) if n > 1 and span > 0 else 0.5  # tegrastats --interval 500

    busy = [x.watts for x in s if x.gpu_pct > 0]
    first_busy = next((i for i, x in enumerate(s) if x.gpu_pct > 0), None)
    if first_busy:
        p_idle = sum(x.watts for x in s[:first_busy]) / first_busy
    else:  # log starts busy (or never busy): fall back to the quietest sample
        p_idle = min(x.watts for x in s)

    t_run = (warmup + iters) * latency_ms_mean / 1000.0
    e_dyn = sum((x.watts - p_idle) * dt for x in s)
    w_run = p_idle + e_dyn / t_run

    return {
        "W": w_run,
        "mJ": w_run * latency_ms_mean,
        "W_idle": p_idle,
        "W_busy": sum(busy) / len(busy) if busy else float("nan"),
        "W_logmean": sum(x.watts for x in s) / n,
        "n": n,
        "n_busy": len(busy),
        "dt_s": dt,
        "t_run_s": t_run,
        "t_busy_s": len(busy) * dt,
    }


def energy_for_run(run_dir: str | Path, warmup: int = 20) -> dict:
    """A results directory holding bench.json and tegrastats.log."""
    run_dir = Path(run_dir)
    bench = json.loads((run_dir / "bench.json").read_text())
    r = energy_from_log(run_dir / "tegrastats.log", bench["latency_ms_mean"], bench["iters"], warmup)
    r["ms"] = bench["latency_ms_mean"]
    return r


_MODES = {"m0_15W": "15W", "m1_25W": "25W", "m2_MAXN_SUPER": "MAXN_SUPER"}


def rebuild_sweep(sweep_dir: str | Path) -> list[dict]:
    """Recompute summary.json for the power-mode sweep from its raw logs.

    Keeps the original whole-log mean under W_logmean / mJ_logmean.
    """
    sweep_dir = Path(sweep_dir)
    rows = []
    for mode_dir in sorted(sweep_dir.iterdir()):
        if not mode_dir.is_dir():
            continue
        prefix, state = mode_dir.name.rsplit("_", 1)
        for run in sorted(mode_dir.iterdir()):
            if not (run / "bench.json").exists():
                continue
            bench = json.loads((run / "bench.json").read_text())
            e = energy_for_run(run)
            rows.append({
                "mode": _MODES[prefix], "state": state, "v": run.name.removeprefix("trt_"),
                "ms": bench["latency_ms_mean"], "p95": bench["latency_ms_p95"],
                "fps": bench["throughput_img_s"],
                "W": e["W"], "mJ": e["mJ"], "W_busy": e["W_busy"], "W_idle": e["W_idle"],
                "W_logmean": e["W_logmean"], "mJ_logmean": e["W_logmean"] * bench["latency_ms_mean"],
                "n": e["n"], "n_busy": e["n_busy"],
            })
    (sweep_dir / "summary.json").write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="result dirs with bench.json + tegrastats.log")
    ap.add_argument("--rebuild-sweep", metavar="SWEEP_DIR", help="rewrite <SWEEP_DIR>/summary.json")
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    hdr = f"{'run':44s} {'ms':>7s} {'W':>6s} {'mJ':>6s} {'W_busy':>7s} {'W_idle':>7s} {'W_log':>6s} {'busy':>5s}"
    if args.rebuild_sweep:
        rows = rebuild_sweep(args.rebuild_sweep)
        print(hdr)
        for r in rows:
            name = f"{r['mode']}/{r['state']}/{r['v']}"
            print(f"{name:44s} {r['ms']:7.2f} {r['W']:6.2f} {r['mJ']:6.0f} {r['W_busy']:7.2f} "
                  f"{r['W_idle']:7.2f} {r['W_logmean']:6.2f} {r['n_busy']:3d}/{r['n']}")
    if args.runs:
        print(hdr)
        for d in args.runs:
            e = energy_for_run(d, args.warmup)
            print(f"{d:44s} {e['ms']:7.2f} {e['W']:6.2f} {e['mJ']:6.0f} {e['W_busy']:7.2f} "
                  f"{e['W_idle']:7.2f} {e['W_logmean']:6.2f} {e['n_busy']:3d}/{e['n']}")


if __name__ == "__main__":
    main()
