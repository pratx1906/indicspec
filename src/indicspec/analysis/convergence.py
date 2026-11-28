"""Statistical convergence check over every speed measurement in the repo (exploratory, D-028).

For each (run, arm, language, concurrency) it asks two questions about the paired speed-up and the
acceptance length tau:

* **Precision:** is the 95% paired-bootstrap CI half-width at most ``target`` (default 5%) of the
  estimate? If not, how many prompts would reach it (CI width shrinks with 1/sqrt(n))?
* **Stability:** does the running estimate settle? The prompts are put in a fixed random order and
  the estimate is recomputed on the first 50%, 60%, ..., 100% of them; ``drift`` is the largest
  relative distance from the full-sample estimate over that range.

The study's conclusions are contrasts between languages, so it also bootstraps the Dravidian/English
speed-up ratio over items (items are parallel across languages, so they are resampled jointly), and
reports whether that CI excludes 1.

Sources: engine sweeps under results/raw/<run>/ (requests.jsonl, per-prompt tokens/s = output
tokens / end-to-end latency, as in indicspec.analysis.pilot) and the cross-model quick test
(results/raw/quick_models/speed*.jsonl, one user, A0 vs n-gram).

Usage: python -m indicspec.analysis.convergence [--target 0.05]
Outputs: results/tables/convergence.csv, results/tables/convergence_contrasts.csv, printed summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from indicspec.rawio import read_jsonl

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "results/raw"
TABLES = REPO / "results/tables"
DRAV = ("ta", "te", "kn", "ml")
SWEEPS = (
    "confirm_vllm_4b",
    "confirm_vllm_8b",
    "explore_sampled_vllm_4b",
    "explore_long_vllm_4b",
    "explore_thinking_vllm_4b",
    "pilot_vllm_4b",
    "pilot_vllm_8b",
    "xengine_sglang_4b_flush",
)


def boot_ratio(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 1234
) -> tuple[float, float, float]:
    """Ratio of means sum(a)/sum(b) with a paired percentile-bootstrap 95% CI."""
    xa = np.asarray(a, dtype=np.float64)
    xb = np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, xa.size, size=(n_boot, xa.size))
    boots = xa[idx].sum(1) / xb[idx].sum(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(xa.sum()) / float(xb.sum()), float(lo), float(hi)


def drift(a: Sequence[float], b: Sequence[float], seed: int = 1234) -> float:
    """Largest relative gap between the running ratio (50%..100% of items) and the full estimate."""
    xa = np.asarray(a, dtype=np.float64)
    xb = np.asarray(b, dtype=np.float64)
    order = np.random.default_rng(seed).permutation(xa.size)
    xa, xb = xa[order], xb[order]
    full = float(xa.sum()) / float(xb.sum())
    ks = sorted({max(2, math.ceil(f * xa.size)) for f in np.arange(0.5, 1.0001, 0.1)})
    return float(max(abs(float(xa[:k].sum()) / float(xb[:k].sum()) - full) / full for k in ks))


def needed(n: int, rel_hw: float, target: float) -> int:
    return n if rel_hw <= target else math.ceil(n * (rel_hw / target) ** 2)


def _tps(r: dict[str, Any]) -> float | None:
    e2e = r.get("e2e_latency_s", r.get("latency_s"))
    if not e2e or r.get("error") or not r.get("output_tokens"):
        return None
    return float(r["output_tokens"]) / float(e2e)


def sweep_groups(run: str) -> dict[tuple[str, str, int], dict[str, dict[str, list[float]]]]:
    """(arm, lang, conc) -> prompt_id -> {tps, acc, steps} lists over repeats."""
    out: dict[tuple[str, str, int], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for block in sorted((RAW / run).glob("*/")):
        f = block / "requests.jsonl"
        if not (f.exists() or f.with_suffix(".jsonl.gz").exists()):
            continue
        for r in read_jsonl(f):
            v = _tps(r)
            if v is None:
                continue
            d = out[(r["arm"], r["language"], int(r["concurrency"]))][r["prompt_id"]]
            d["tps"].append(v)
            d["acc"].append(float(r.get("spec_accepted_tokens") or 0))
            d["steps"].append(float(r.get("spec_verify_steps") or 0))
    return out


def quick_groups() -> dict[tuple[str, str, str, int], dict[str, dict[str, list[float]]]]:
    out: dict[tuple[str, str, str, int], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for name, eng in (("speed.jsonl", "vllm"), ("speed_hf.jsonl", "transformers")):
        f = RAW / "quick_models" / name
        if not f.exists():
            continue
        for r in read_jsonl(f):
            # the first, interrupted Tamil-LLaMA run is superseded (D-025)
            if r["model"] == "tamil-llama-7b" and r.get("time", "") < "2026-11-27T22:05":
                continue
            v = _tps(r)
            if v is not None:
                key = (f"quick_models[{eng}]:{r['model']}", "A0" if r["arm"] == "A0" else "ngram8")
                out[(*key, r["lang"], 1)][r["item"]]["tps"].append(v)
    return out


def analyse(target: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    sources: list[tuple[str, dict[tuple[str, str, int], Any]]] = []
    for run in SWEEPS:
        if (RAW / run).exists():
            sources.append((run, sweep_groups(run)))
    by_model: dict[str, dict[tuple[str, str, int], Any]] = defaultdict(dict)
    for (model, arm, lang, conc), g in quick_groups().items():
        by_model[model][(arm, lang, conc)] = g
    sources += sorted(by_model.items())

    for run, groups in sources:
        base = {(lang, c): g for (arm, lang, c), g in groups.items() if arm == "A0"}
        per_item: dict[tuple[str, int], dict[str, dict[str, float]]] = defaultdict(dict)
        for (arm, lang, conc), g in sorted(groups.items()):
            if arm == "A0" or (lang, conc) not in base:
                continue
            b = base[(lang, conc)]
            ids = sorted(set(g) & set(b))
            if len(ids) < 3:
                continue
            a_t = [float(np.mean(g[i]["tps"])) for i in ids]
            b_t = [float(np.mean(b[i]["tps"])) for i in ids]
            est, lo, hi = boot_ratio(a_t, b_t)
            rhw = (hi - lo) / 2 / est
            reps = max(len(g[i]["tps"]) for i in ids)
            row: dict[str, Any] = {
                "run": run,
                "arm": arm,
                "language": lang,
                "concurrency": conc,
                "n_items": len(ids),
                "repeats": reps,
                "speedup": est,
                "ci_lo": lo,
                "ci_hi": hi,
                "rel_half_width": rhw,
                "drift": drift(a_t, b_t),
                "items_needed": needed(len(ids), rhw, target),
            }
            steps = [sum(g[i]["steps"]) for i in ids]
            if all(steps):
                num = [sum(g[i]["acc"]) + sum(g[i]["steps"]) for i in ids]
                t, tlo, thi = boot_ratio(num, steps)
                row.update(tau=t, tau_rel_half_width=(thi - tlo) / 2 / t)
            row["converged"] = rhw <= target and row["drift"] <= target
            rows.append(row)
            for i, x, y in zip(ids, a_t, b_t, strict=True):
                per_item[(arm, conc)].setdefault(i, {})[lang] = x / y
        # Dravidian / English contrast, items resampled jointly across languages
        for (arm, conc), items in sorted(per_item.items()):
            full = [v for v in items.values() if "en" in v and all(x in v for x in DRAV)]
            if len(full) < 3:
                continue
            m = np.array([[v["en"], *(v[x] for x in DRAV)] for v in full])
            rng = np.random.default_rng(1234)
            idx = rng.integers(0, len(m), size=(2000, len(m)))
            sub = m[idx]  # (boot, n, 5)
            boots = sub[:, :, 1:].mean(axis=(1, 2)) / sub[:, :, 0].mean(axis=1)
            est = float(m[:, 1:].mean() / m[:, 0].mean())
            lo, hi = (float(q) for q in np.quantile(boots, [0.025, 0.975]))
            contrasts.append(
                {
                    "run": run,
                    "arm": arm,
                    "concurrency": conc,
                    "n_items": len(full),
                    "dravidian_over_english": est,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "excludes_1": lo > 1 or hi < 1,
                }
            )
    return rows, contrasts


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.05, help="relative CI half-width target")
    a = ap.parse_args(argv)
    rows, contrasts = analyse(a.target)
    _write(TABLES / "convergence.csv", rows)
    _write(TABLES / "convergence_contrasts.csv", contrasts)
    print(f"target: 95% CI half-width <= {a.target:.0%} of the estimate, running drift <= same")
    print(f"{'source':48} cells  converged  worst_rel_hw  max_items_needed")
    for run in dict.fromkeys(r["run"] for r in rows):
        rs = [r for r in rows if r["run"] == run]
        ok = sum(r["converged"] for r in rs)
        print(
            f"{run:48} {len(rs):5}  {ok:9}  {max(r['rel_half_width'] for r in rs):12.3f}"
            f"  {max(r['items_needed'] for r in rs):16}"
        )
    bad = [c for c in contrasts if not c["excludes_1"]]
    print(
        f"\nlanguage contrasts (Dravidian/English speed-up): {len(contrasts)} tested, "
        f"{len(contrasts) - len(bad)} with CI excluding 1"
    )
    for c in bad:
        print(
            f"  CI includes 1: {c['run']} {c['arm']} c{c['concurrency']} "
            f"{c['dravidian_over_english']:.2f} [{c['ci_lo']:.2f}, {c['ci_hi']:.2f}]"
        )
    print(json.dumps({"rows": len(rows), "contrasts": len(contrasts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
