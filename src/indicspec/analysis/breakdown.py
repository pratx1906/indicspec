"""Exploratory breakdowns of a confirmatory run (not pre-registered; never used for verdicts).

Reads the same raw logs as ``confirm.py`` and writes three tables:

  results/tables/<run>_by_task.csv    speed-up and tau per (arm, language, task), concurrency 1
  results/tables/<run>_energy.csv     GPU energy per 1k output tokens and per 100 output words,
                                      per (arm, language, concurrency), and relative to A0
  results/tables/<run>_class_acceptance.csv
                                      draft acceptance per Qwen3 token class at evaluated
                                      positions (engine per-step traces), concurrency 1

Speed-up = mean over items of the arm's per-request tokens/s (mean of repeats) / the same for A0,
paired by prompt id (the PREREG definition, restricted to one task). Energy comes from the NVML
total-energy counter per cell (``cells.jsonl``); it covers the whole GPU, idle gaps included.

Usage (WSL; needs the tokenizer in the HF cache):
  python -m indicspec.analysis.breakdown --run confirm_vllm_4b
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
LANGS = ["en", "hi", "ta", "te", "kn", "ml"]


def _jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _tps(r: Mapping[str, Any]) -> float | None:
    if r.get("error") or not r.get("e2e_latency_s") or not r.get("output_tokens"):
        return None
    return float(r["output_tokens"]) / float(r["e2e_latency_s"])


def by_task(rows: Sequence[Mapping[str, Any]], conc: int = 1) -> list[dict[str, Any]]:
    """Speed-up vs A0 and tau per (arm, language, task), paired by prompt id."""
    tps: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    acc: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        if int(r["concurrency"]) != conc:
            continue
        v = _tps(r)
        if v is not None:
            tps[(r["arm"], r["language"], r["task"], r["prompt_id"])].append(v)
        if r.get("spec_verify_steps"):
            a = acc[(r["arm"], r["language"], r["task"])]
            a[0] += int(r.get("spec_accepted_tokens") or 0)
            a[1] += int(r["spec_verify_steps"])
    mean = {k: sum(v) / len(v) for k, v in tps.items()}
    arms = sorted({k[0] for k in mean} - {"A0"})
    tasks = sorted({k[2] for k in mean})
    out = []
    for arm in arms:
        for lang in LANGS:
            for task in tasks:
                pairs = [
                    (v, mean[("A0", lang, task, pid)])
                    for (a, lg, t, pid), v in mean.items()
                    if a == arm and lg == lang and t == task and ("A0", lang, task, pid) in mean
                ]
                if not pairs:
                    continue
                s = acc.get((arm, lang, task), [0, 0])
                out.append(
                    {
                        "arm": arm,
                        "language": lang,
                        "task": task,
                        "n_items": len(pairs),
                        "speedup": (sum(p[0] for p in pairs) / sum(p[1] for p in pairs)),
                        "tau": (s[0] + s[1]) / s[1] if s[1] else None,
                    }
                )
    return out


def energy(
    cells: Iterable[tuple[str, Mapping[str, Any]]], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Joules per 1k output tokens and per 100 output words per (arm, language, concurrency)."""
    words: dict[tuple[str, str, int, int], int] = defaultdict(int)
    for r in rows:
        if not r.get("error"):
            key = (r["arm"], r["language"], int(r["concurrency"]), int(r["repeat"]))
            words[key] += int(r.get("output_words") or 0)
    agg: dict[tuple[str, str, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for arm, c in cells:
        if c.get("energy_mj") is None:
            continue
        lang, conc, rep = c["language"], int(c["concurrency"]), int(c["repeat"])
        a = agg[(arm, lang, conc)]
        a[0] += float(c["energy_mj"]) / 1000.0  # mJ -> J
        a[1] += float(c["total_output_tokens"])
        a[2] += words[(arm, lang, conc, rep)]
    base = {(lg, cc): v[0] / v[1] for (arm, lg, cc), v in agg.items() if arm == "A0" and v[1]}
    out = []
    for (arm, lang, conc), (j, toks, w) in sorted(agg.items()):
        if not toks:
            continue
        per_tok = j / toks
        out.append(
            {
                "arm": arm,
                "language": lang,
                "concurrency": conc,
                "joules_per_1k_tokens": 1000 * per_tok,
                "joules_per_100_words": 100 * j / w if w else None,
                "energy_vs_A0": per_tok / base[(lang, conc)] if (lang, conc) in base else None,
            }
        )
    return out


def class_acceptance(
    rows: Sequence[Mapping[str, Any]], token_class: Mapping[int, str], conc: int = 1
) -> list[dict[str, Any]]:
    """Acceptance rate per token class at evaluated positions, per (arm, language)."""
    from indicspec.analysis.confirm import positions_for

    counts: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    arms = sorted({r["arm"] for r in rows} - {"A0"})
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm and int(r["concurrency"]) == conc]
        for lang, ps in positions_for(sel).per_request:
            for tok, ok in ps:
                c = counts[(arm, lang, token_class.get(tok, "UNKNOWN"))]
                c[0] += int(ok)
                c[1] += 1
    return [
        {
            "arm": arm,
            "language": lang,
            "token_class": cls,
            "n_positions": n,
            "share_of_positions": n / sum(v[1] for k, v in counts.items() if k[:2] == (arm, lang)),
            "acceptance_rate": a / n,
        }
        for (arm, lang, cls), (a, n) in sorted(counts.items())
        if n
    ]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-classes", action="store_true", help="skip the tokenizer-based table")
    a = ap.parse_args(argv)
    run_dir = REPO / "results/raw" / a.run
    rows: list[dict[str, Any]] = []
    cells: list[tuple[str, dict[str, Any]]] = []
    for block in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        br = [r for r in _jsonl(block / "requests.jsonl") if r.get("split") == "report"]
        rows += br
        arm = br[0]["arm"] if br else None
        if arm:
            cells += [(arm, c) for c in _jsonl(block / "cells.jsonl")]
    out = REPO / "results/tables"
    _write(out / f"{a.run}_by_task.csv", by_task(rows))
    _write(out / f"{a.run}_energy.csv", energy(cells, rows))
    if not a.no_classes:
        from indicspec.analysis.confirm import _load_token_classes

        _write(out / f"{a.run}_class_acceptance.csv", class_acceptance(rows, _load_token_classes()))
    print(
        f"wrote {a.run}_by_task.csv, _energy.csv"
        + ("" if a.no_classes else ", _class_acceptance.csv")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
