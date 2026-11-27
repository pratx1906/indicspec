"""Cross-engine check tables (exploratory; D-023, C57-C58). Not part of the confirmatory analysis.

Compares SGLang runs on the tuning split against the vLLM tuning grid for the same arms and
prompts. Speed-up at concurrency 1 and tau come from ``tuning.grid``; concurrency-16 speed-up is the
paired ratio of summed per-request tokens/s (one repeat per cell in these runs).

  results/tables/xengine_sglang_4b.csv        unflushed SGLang run (A1, A5) vs vLLM
  results/tables/xengine_sglang_4b_flush.csv  A1 with the NGRAM corpus flushed per cell, vs both

Usage: python -m indicspec.analysis.xengine
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from indicspec.analysis.tuning import grid
from indicspec.rawio import read_jsonl

REPO = Path(__file__).resolve().parents[3]
RAW, TABLES = REPO / "results/raw", REPO / "results/tables"
LANGS = ["en", "hi", "ta", "te", "kn", "ml"]
# SGLang arm label -> the vLLM tuning setting it was configured to match
ARM_MAP = {"A1": "A1.k8", "A5": "A5.k15"}


def _tps(r: dict[str, Any]) -> float | None:
    if r.get("error") or not r.get("e2e_latency_s"):
        return None
    return float(r["output_tokens"]) / float(r["e2e_latency_s"])


def speedup_at(run_dir: Path, conc: int) -> dict[tuple[str, str], float]:
    """Paired speed-up vs A0 per (arm, language) at one concurrency."""
    m: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for block in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for r in read_jsonl(block / "requests.jsonl"):
            v = _tps(r)
            if int(r["concurrency"]) == conc and v:
                m[(r["arm"], r["language"])][r["prompt_id"]] = v
    out = {}
    for (arm, lang), per in m.items():
        if arm == "A0" or ("A0", lang) not in m:
            continue
        a0 = m[("A0", lang)]
        ks = [k for k in per if k in a0]
        out[(arm, lang)] = sum(per[k] for k in ks) / sum(a0[k] for k in ks)
    return out


def _grid(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["arm"], r["language"]): r for r in grid(run_dir)}


def _vllm(name: str) -> dict[tuple[str, str], dict[str, str]]:
    with open(TABLES / name, encoding="utf-8") as fh:
        return {(r["setting"], r["language"]): r for r in csv.DictReader(fh)}


def unflushed(run: str = "xengine_sglang_4b", vgrid: str = "tune_vllm_4b_grid.csv") -> list[dict]:
    sg, vl, c16 = _grid(RAW / run), _vllm(vgrid), speedup_at(RAW / run, 16)
    out = []
    for arm, vset in ARM_MAP.items():
        for lang in LANGS:
            s, v = sg[(arm, lang)], vl[(vset, lang)]
            out.append(
                {
                    "arm": vset,
                    "language": lang,
                    "sglang_speedup_c1": s["speedup"],
                    "vllm_speedup_c1": float(v["speedup"]),
                    "sglang_tau_c1": s["tau"],
                    "vllm_tau_c1": float(v["tau"]),
                    "sglang_speedup_c16": c16[(arm, lang)],
                    "n_items": s["n_pairs"],
                }
            )
    return out


def flushed(
    run: str = "xengine_sglang_4b_flush",
    noflush: str = "xengine_sglang_4b",
    vgrid: str = "tune_vllm_4b_grid.csv",
) -> list[dict]:
    fl, nf, vl = _grid(RAW / run), _grid(RAW / noflush), _vllm(vgrid)
    c16 = speedup_at(RAW / run, 16)
    return [
        {
            "language": lang,
            "sglang_flush_speedup_c1": fl[("A1", lang)]["speedup"],
            "sglang_flush_tau_c1": fl[("A1", lang)]["tau"],
            "sglang_flush_speedup_c16": c16[("A1", lang)],
            "sglang_noflush_speedup_c1": nf[("A1", lang)]["speedup"],
            "sglang_noflush_tau_c1": nf[("A1", lang)]["tau"],
            "vllm_speedup_c1": float(vl[("A1.k8", lang)]["speedup"]),
            "vllm_tau_c1": float(vl[("A1.k8", lang)]["tau"]),
        }
        for lang in LANGS
    ]


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{x:.4f}" if isinstance(x, float) else x) for k, x in r.items()})


def main() -> int:
    _write(TABLES / "xengine_sglang_4b.csv", unflushed())
    _write(TABLES / "xengine_sglang_4b_flush.csv", flushed())
    print("wrote xengine_sglang_4b.csv, xengine_sglang_4b_flush.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
