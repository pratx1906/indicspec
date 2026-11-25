from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from indicspec import env_probe
from indicspec.harness.schema import validate
from indicspec.harness.telemetry import summarise

GOOD = {
    "schema_version": "1",
    "run_id": "r1",
    "arm": "A0",
    "engine": "vllm",
    "model": "Qwen/Qwen3-4B",
    "language": "ta",
    "task": "translation",
    "prompt_id": "flores-0001",
    "split": "smoke",
    "concurrency": 1,
    "load_mode": "closed_loop",
    "repeat": 0,
    "input_tokens": 10,
    "output_tokens": 20,
    "ttft_s": 0.05,
    "e2e_latency_s": 0.5,
    "error": None,
}


def test_request_log_schema_accepts_good_row():
    validate(GOOD, "request_log.v1")


@pytest.mark.parametrize(
    "patch",
    [{"language": "fr"}, {"concurrency": 0}, {"arm": "B1"}, {"unexpected": 1}],
)
def test_request_log_schema_rejects_bad_rows(patch):
    with pytest.raises(jsonschema.ValidationError):
        validate({**GOOD, **patch}, "request_log.v1")


def test_parse_nvsmi_csv():
    rows = env_probe.parse_nvsmi_csv(
        "NVIDIA GeForce RTX 5090 Laptop GPU, 24463 MiB, 610.88, 12.0\n",
        ["name", "mem", "drv", "cc"],
    )
    assert rows == [
        {
            "name": "NVIDIA GeForce RTX 5090 Laptop GPU",
            "mem": "24463 MiB",
            "drv": "610.88",
            "cc": "12.0",
        }
    ]


def test_probe_writes_valid_json(tmp_path: Path):
    out = tmp_path / "probe.json"
    assert env_probe.main(["--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == "indicspec.env_probe/1"
    assert {"os", "gpu", "host", "nvcc", "torch"} <= data.keys()


def _write_csv(p: Path, temps, clocks):
    lines = [
        "timestamp, clocks.sm [MHz], clocks.mem [MHz], power.draw [W], temperature.gpu, "
        "memory.used [MiB], utilization.gpu [%], clocks_event_reasons.active"
    ]
    for t, c in zip(temps, clocks, strict=True):
        lines.append(f"2026/11/25 10:00:00.000, {c}, 12000, 100.0, {t}, 1000, 90, 0x0")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_thermal_summary_reads_canonical_clock_header(tmp_path: Path):
    """Header exactly as written by nvidia-smi 610.88 (observed in a real smoke run)."""
    p = tmp_path / "t.csv"
    p.write_text(
        "timestamp, clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W], "
        "temperature.gpu, memory.used [MiB], utilization.gpu [%], clocks_event_reasons.active\n"
        "2026/11/25 18:27:56.257, 1597, 14001, 33.15, 65, 1636, 0, 0x0000000000000000\n",
        encoding="utf-8",
    )
    s = summarise(p, max_temp_c=87, min_sm_clock_ratio=0.85)
    assert s.median_sm_clock == 1597 and s.peak_mem_mib == 1636


def test_thermal_summary_ok_and_breach(tmp_path: Path):
    p = tmp_path / "t.csv"
    _write_csv(p, [70] * 50, [2500] * 50)
    s = summarise(p, max_temp_c=87, min_sm_clock_ratio=0.85)
    assert s.ok and s.n_samples == 50
    _write_csv(p, [70] * 40 + [90] * 10, [2500] * 40 + [1500] * 10)
    s = summarise(p, max_temp_c=87, min_sm_clock_ratio=0.85)
    assert not s.ok and len(s.breaches) == 2
