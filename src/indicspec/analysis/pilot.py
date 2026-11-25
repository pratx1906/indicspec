"""Pilot noise analysis -> inputs for docs/PREREG.md thresholds (spec §5, §8.4).

Reads ``results/raw/<run>/<block>/{requests.jsonl, cells.jsonl, telemetry-*.csv}`` and writes
``results/tables/<run>_noise.csv`` with, per (arm, language, concurrency):

* n_prompts, n_repeats, errors
* tau: mean, between-repeat SD of the cell mean, within-cell prompt SD, bootstrap 95% CI
  half-width at the pilot n, and the prompts needed for a target half-width (``--tau-hw``)
* per-request tokens/s: mean, between-repeat CV, paired speed-up vs A0 (same prompts/cell) with
  a paired bootstrap CI
* system throughput (tokens / wall), energy per output token (J) and per 100 output words

and ``results/tables/<run>_thermal.csv`` per block (max temperature, SM clock percentiles,
peak VRAM). Everything here is **pilot / exploratory**; it is used only to set thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from indicspec.analysis.stats import bootstrap_ci, paired_bootstrap_ci
from indicspec.harness.telemetry import summarise

REPO = Path(__file__).resolve().parents[3]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _tau(rs: list[dict[str, Any]]) -> float | None:
    a = sum(r.get("spec_accepted_tokens") or 0 for r in rs)
    s = sum(r.get("spec_verify_steps") or 0 for r in rs)
    return (a + s) / s if s else None


def _tps(r: dict[str, Any]) -> float | None:
    e2e = r.get("e2e_latency_s")
    if not e2e or r.get("error") or not r.get("output_tokens"):
        return None
    return float(r["output_tokens"]) / float(e2e)


def needed_n(sd: float, half_width: float) -> int | None:
    """Prompts needed so a normal-approx 95% CI of a mean has the given half-width."""
    if not sd or half_width <= 0 or math.isnan(sd):
        return None
    return math.ceil((1.96 * sd / half_width) ** 2)


def analyse(run_dir: Path, tau_hw: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reqs: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    thermal: list[dict[str, Any]] = []
    for block in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        rr = _rows(block / "requests.jsonl")
        cc = _rows(block / "cells.jsonl")
        arm = rr[0]["arm"] if rr else block.name
        for c in cc:
            c["arm"] = arm
        reqs += rr
        cells += cc
        for t in sorted(block.glob("telemetry-*.csv")):
            try:
                s = summarise(t, max_temp_c=1e9, min_sm_clock_ratio=0.0, skip_first_s=0)
            except ValueError:
                continue
            thermal.append(
                {
                    "block": block.name,
                    "file": t.name,
                    "n_samples": s.n_samples,
                    "max_temp_c": s.max_temp_c,
                    "median_sm_clock": s.median_sm_clock,
                    "min_sm_clock": s.min_sm_clock,
                    "max_power_w": s.max_power_w,
                    "peak_mem_mib": s.peak_mem_mib,
                }
            )

    by: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in reqs:
        by[(r["arm"], r["language"], int(r["concurrency"]))].append(r)
    base_tps: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for (arm, lang, conc), rs in by.items():
        if arm == "A0":
            for r in rs:
                v = _tps(r)
                if v is not None:
                    base_tps[(lang, conc, r["prompt_id"])].append(v)

    cell_by: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for c in cells:
        cell_by[(c["arm"], c["language"], int(c["concurrency"]))].append(c)

    out = []
    for (arm, lang, conc), rs in sorted(by.items()):
        reps = sorted({int(r["repeat"]) for r in rs})
        rep_tau = [t for t in (_tau([r for r in rs if r["repeat"] == k]) for k in reps) if t]
        prompt_tau: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            if r.get("acceptance_length"):
                prompt_tau[r["prompt_id"]].append(float(r["acceptance_length"]))
        ptau = [float(np.mean(v)) for v in prompt_tau.values()]
        tps_by_prompt: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            v = _tps(r)
            if v is not None:
                tps_by_prompt[r["prompt_id"]].append(v)
        rep_tps = [
            float(np.mean([v for r in rs if r["repeat"] == k and (v := _tps(r)) is not None]))
            for k in reps
            if any(_tps(r) is not None for r in rs if r["repeat"] == k)
        ]
        row: dict[str, Any] = {
            "arm": arm,
            "language": lang,
            "concurrency": conc,
            "n_prompts": len({r["prompt_id"] for r in rs}),
            "n_repeats": len(reps),
            "errors": sum(1 for r in rs if r.get("error")),
            "tau": _tau(rs),
            "tau_sd_between_repeats": float(np.std(rep_tau, ddof=1)) if len(rep_tau) > 1 else None,
            "tau_sd_prompts": float(np.std(ptau, ddof=1)) if len(ptau) > 1 else None,
            "tps_mean": float(np.mean([v for vs in tps_by_prompt.values() for v in vs]))
            if tps_by_prompt
            else None,
            "tps_cv_between_repeats": float(np.std(rep_tps, ddof=1) / np.mean(rep_tps))
            if len(rep_tps) > 1
            else None,
        }
        if len(ptau) > 1:
            ci = bootstrap_ci(ptau, n_boot=2000)
            row["tau_ci95_halfwidth"] = (ci.hi - ci.lo) / 2
            row["prompts_needed_for_tau_hw"] = needed_n(row["tau_sd_prompts"], tau_hw)
        if arm != "A0":
            pairs = [
                (float(np.mean(v)), float(np.mean(base_tps[(lang, conc, pid)])))
                for pid, v in tps_by_prompt.items()
                if base_tps.get((lang, conc, pid))
            ]
            if len(pairs) > 1:
                a = [p[0] for p in pairs]
                b = [p[1] for p in pairs]
                sp = paired_bootstrap_ci(a, b, n_boot=2000)
                row.update(speedup=sp.estimate, speedup_ci_lo=sp.lo, speedup_ci_hi=sp.hi)
        cs = cell_by.get((arm, lang, conc), [])
        tot_tok = sum(c.get("total_output_tokens") or 0 for c in cs)
        wall = sum(c.get("wall_s") or 0 for c in cs)
        energy = [c["energy_mj"] for c in cs if c.get("energy_mj") is not None]
        row["system_tps"] = tot_tok / wall if wall else None
        row["joules_per_token"] = (sum(energy) / 1000 / tot_tok) if energy and tot_tok else None
        words = sum(r.get("output_words") or 0 for r in rs)
        row["joules_per_100_words"] = (
            sum(energy) / 1000 / (words / 100) if energy and words else None
        )
        out.append(row)
    return out, thermal


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/raw/<run>")
    ap.add_argument("--tau-hw", type=float, default=0.1, help="target 95%% CI half-width for tau")
    a = ap.parse_args(argv)
    run_dir = REPO / "results/raw" / a.run
    rows, thermal = analyse(run_dir, a.tau_hw)
    out = REPO / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    _write(out / f"{a.run}_noise.csv", rows)
    _write(out / f"{a.run}_thermal.csv", thermal)
    print(f"{len(rows)} cells, {len(thermal)} telemetry files -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
