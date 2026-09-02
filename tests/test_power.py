import json

from segdeploy.power import energy_for_run, energy_from_log, parse_tegrastats


def _line(sec: int, mw: int, gpu: int) -> str:
    return (f"08-22-2026 00:00:{sec:02d} RAM 1/1MB CPU [0%@1] EMC_FREQ 1%@1 "
            f"GR3D_FREQ {gpu}%@[1000] VDD_IN {mw}mW/{mw}mW/{mw}mW")


def _write_log(path, idle_mw=6000, busy_mw=16000, n_idle=4, n_busy=10):
    # one sample per second so dt is exactly 1 s and the arithmetic is exact
    lines, i = [], 0
    for _ in range(n_idle):
        lines.append(_line(i, idle_mw, 0)); i += 1
    for _ in range(n_busy):
        lines.append(_line(i, busy_mw, 80)); i += 1
    for _ in range(n_idle):
        lines.append(_line(i, idle_mw, 0)); i += 1
    path.write_text("\n".join(lines) + "\n")


def test_parse(tmp_path):
    _write_log(tmp_path / "t.log")
    s = parse_tegrastats(tmp_path / "t.log")
    assert len(s) == 18 and s[0].watts == 6.0 and s[4].gpu_pct == 80


def test_idle_is_excluded(tmp_path):
    # 10 busy samples at 1 s = 10 s run of 220 iterations -> 45.5 ms each.
    # Idle padding must not pull the answer below the 16 W plateau.
    _write_log(tmp_path / "t.log")
    ms = 10000 / 220
    e = energy_from_log(tmp_path / "t.log", ms, iters=200, warmup=20)
    assert abs(e["W"] - 16.0) < 1e-6
    assert abs(e["mJ"] - 16.0 * ms) < 1e-6
    assert abs(e["W_busy"] - 16.0) < 1e-6
    assert e["W_logmean"] < 12  # the old whole-log mean, kept for the record
    assert e["W_idle"] == 6.0 and e["n_busy"] == 10


def test_ramp_energy_is_conserved(tmp_path):
    # A lagging sensor smears the step: half the first busy sample's excess
    # shows up one sample after the GPU goes idle. The integral is unchanged.
    log = tmp_path / "t.log"
    _write_log(log)
    lines = log.read_text().splitlines()
    lines[4] = _line(4, 11000, 80)   # first busy sample reads low
    lines[14] = _line(14, 11000, 0)  # first idle sample after the run reads high
    log.write_text("\n".join(lines) + "\n")
    e = energy_from_log(log, 10000 / 220)
    assert abs(e["W"] - 16.0) < 1e-6
    assert e["W_busy"] < 16.0      # the busy-window mean is fooled


def test_energy_for_run_reads_bench(tmp_path):
    _write_log(tmp_path / "tegrastats.log")
    (tmp_path / "bench.json").write_text(json.dumps(
        {"iters": 200, "latency_ms_mean": 10000 / 220, "latency_ms_p95": 23.0, "throughput_img_s": 44.0}))
    e = energy_for_run(tmp_path)
    assert abs(e["W"] - 16.0) < 1e-6 and abs(e["ms"] - 10000 / 220) < 1e-9
