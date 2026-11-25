"""GPU telemetry sampler and thermal-rule evaluation.

Samples clocks, power, temperature and memory once per interval with
``nvidia-smi`` (read-only; changes no GPU settings) into a CSV, and
summarises a CSV against the pre-registered thermal-abort rule.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO

FIELDS = [
    "timestamp",
    "clocks.sm",
    "clocks.mem",
    "power.draw",
    "temperature.gpu",
    "memory.used",
    "utilization.gpu",
    "clocks_event_reasons.active",
]


class TelemetrySampler:
    """Context manager: ``with TelemetrySampler(path): run()``."""

    def __init__(self, out_csv: Path, interval_ms: int = 1000) -> None:
        self.out_csv = Path(out_csv)
        self.interval_ms = interval_ms
        self._proc: subprocess.Popen[bytes] | None = None
        self._fh: IO[bytes] | None = None

    def __enter__(self) -> TelemetrySampler:
        exe = shutil.which("nvidia-smi")
        if exe is None:
            raise RuntimeError("nvidia-smi not found")
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out_csv, "wb")
        self._proc = subprocess.Popen(
            [
                exe,
                f"--query-gpu={','.join(FIELDS)}",
                "--format=csv,nounits",
                f"--loop-ms={self.interval_ms}",
            ],
            stdout=self._fh,
            stderr=subprocess.DEVNULL,
        )
        return self

    def __exit__(self, *exc: object) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._fh is not None:
            self._fh.close()


@dataclass(frozen=True)
class ThermalSummary:
    n_samples: int
    max_temp_c: float
    median_sm_clock: float
    min_sm_clock: float
    max_power_w: float
    peak_mem_mib: float
    breaches: list[str]

    @property
    def ok(self) -> bool:
        return not self.breaches


def _num(x: str) -> float | None:
    try:
        return float(x.strip())
    except (ValueError, AttributeError):
        return None


def summarise(
    csv_path: Path,
    max_temp_c: float,
    min_sm_clock_ratio: float,
    reference_sm_clock: float | None = None,
    skip_first_s: int = 0,
) -> ThermalSummary:
    """Evaluate a telemetry CSV against the thermal rule.

    Breach if any sample's temperature exceeds ``max_temp_c`` or if the
    5th-percentile SM clock falls below ``min_sm_clock_ratio`` x the reference
    (default: the run's own median clock). Thresholds come from docs/PREREG.md.
    """
    temps: list[float] = []
    clocks: list[float] = []
    powers: list[float] = []
    mems: list[float] = []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError("empty telemetry csv")
        col = {h.strip().split(" [")[0]: i for i, h in enumerate(header)}
        for k, row in enumerate(reader):
            if k < skip_first_s or len(row) < len(col):
                continue
            t = _num(row[col["temperature.gpu"]])
            # nvidia-smi echoes the canonical name ("clocks.current.sm") for the "clocks.sm" alias
            c = _num(row[col.get("clocks.sm", col.get("clocks.current.sm", -1))])
            p = _num(row[col["power.draw"]])
            m = _num(row[col["memory.used"]])
            if t is not None:
                temps.append(t)
            if c is not None:
                clocks.append(c)
            if p is not None:
                powers.append(p)
            if m is not None:
                mems.append(m)
    if not clocks:
        raise ValueError("no clock samples")
    s = sorted(clocks)
    median = s[len(s) // 2]
    p5 = s[max(0, int(0.05 * (len(s) - 1)))]
    ref = reference_sm_clock or median
    breaches = []
    if temps and max(temps) > max_temp_c:
        breaches.append(f"temperature {max(temps)} C > {max_temp_c} C")
    if p5 < min_sm_clock_ratio * ref:
        breaches.append(f"p5 SM clock {p5} MHz < {min_sm_clock_ratio} x {ref} MHz")
    return ThermalSummary(
        n_samples=len(clocks),
        max_temp_c=max(temps) if temps else float("nan"),
        median_sm_clock=median,
        min_sm_clock=s[0],
        max_power_w=max(powers) if powers else float("nan"),
        peak_mem_mib=max(mems) if mems else float("nan"),
        breaches=breaches,
    )
