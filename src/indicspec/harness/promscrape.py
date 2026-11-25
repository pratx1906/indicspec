"""Prometheus text-format scraping for engine speculative-decoding counters (WS1.5).

Engines expose cumulative counters (e.g. accepted / drafted tokens). A run's value is the
*difference* between a scrape after and before the run, which is only valid when no other traffic
hits the server in between (the harness owns the server). Exact metric names differ per engine and
version; they live in ``configs/engines.yaml`` once verified against a running server — none are
hard-coded here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import httpx

_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)(?:\s+\d+)?$"
)
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Sample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float


def parse(text: str) -> list[Sample]:
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        labels = tuple(sorted(_LABEL.findall(m.group("labels") or "")))
        v = m.group("value")
        value = {"+Inf": math.inf, "-Inf": -math.inf, "NaN": math.nan}.get(v)
        out.append(Sample(m.group("name"), labels, float(v) if value is None else value))
    return out


def totals(samples: list[Sample], names: list[str]) -> dict[str, float]:
    """Sum each metric over all label sets (e.g. over model / engine index labels)."""
    res = dict.fromkeys(names, 0.0)
    for s in samples:
        if s.name in res:
            res[s.name] += s.value
    return res


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    d = {k: after[k] - before.get(k, 0.0) for k in after}
    bad = {k: v for k, v in d.items() if v < 0}
    if bad:
        raise ValueError(f"counter decreased (server restarted?): {bad}")
    return d


def scrape(base_url: str, names: list[str], timeout: float = 10.0) -> dict[str, float]:
    r = httpx.get(f"{base_url}/metrics", timeout=timeout)
    r.raise_for_status()
    return totals(parse(r.text), names)
