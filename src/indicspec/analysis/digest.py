"""Plain-language digest of a confirmatory analysis, as a draft for CLAIMS/README (not a result).

Reads ``results/tables/<name>_verdicts.json`` and ``<name>_confirm.csv`` (written by confirm.py)
and prints Markdown: every hypothesis and gate verdict with its per-pair and per-language outcomes,
and a speed-up table (one user, 95% CI) per model. Numbers are copied, never recomputed, so the
digest cannot disagree with the analysis.

Usage: python -m indicspec.analysis.digest --name confirm_vllm_4b+confirm_vllm_8b [--out file.md]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
LANGS = ["en", "hi", "ta", "te", "kn", "ml"]
ORDER = ["H1", "H2", "H3", "H4", "G1", "G2", "G3"]


def _short(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
        return "[" + ", ".join(_short(x) for x in v) + "]"
    if isinstance(v, dict):
        return ", ".join(f"{k} {_short(x)}" for k, x in v.items())
    return str(v)


def verdict_lines(d: dict[str, Any]) -> list[str]:
    out = [
        f"PREREG commit {d.get('prereg_commit')}; requests kept {d.get('n_requests_kept')}; "
        f"excluded cells {len(d.get('excluded_cells') or [])}.",
        "",
    ]
    for key in ORDER:
        h = d.get(key)
        if h is None:
            continue
        if not isinstance(h, dict):
            out.append(f"- **{key}**: {h}")
            continue
        overall = h.get("verdict", h.get("gate"))
        out.append(f"- **{key}**: {overall}")
        for pair, p in (h.get("pairs") or {}).items():
            bits = []
            for f in ("verdict", "hit", "a", "b", "per_language", "crossover", "diff_ci"):
                if f in p:
                    bits.append(f"{f} {_short(p[f])}")
            out.append(f"  - {pair}: " + "; ".join(bits))
    return out


def speedup_table(rows: list[dict[str, str]], model: str) -> list[str]:
    arms = sorted({r["arm"] for r in rows if r["model"] == model} - {"A0"})
    if not arms:
        return []
    out = [
        f"Speed-up vs no speculation, one user, {model} (95% CI):",
        "",
        "| Arm | " + " | ".join(LANGS) + " |",
        "|---|" + "---|" * len(LANGS),
    ]
    for arm in arms:
        cells = []
        for lang in LANGS:
            r = next(
                (
                    x
                    for x in rows
                    if x["model"] == model
                    and x["arm"] == arm
                    and x["language"] == lang
                    and x["concurrency"] == "1"
                ),
                None,
            )
            if r is None or not r["speedup"]:
                cells.append("–")
            else:
                s, lo, hi = (float(r[k]) for k in ("speedup", "speedup_ci_lo", "speedup_ci_hi"))
                cells.append(f"{s:.2f} ({lo:.2f}–{hi:.2f})")
        out.append(f"| {arm} | " + " | ".join(cells) + " |")
    return [*out, ""]


def digest(name: str) -> str:
    tables = REPO / "results/tables"
    d = json.loads((tables / f"{name}_verdicts.json").read_text(encoding="utf-8"))
    with open(tables / f"{name}_confirm.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    lines = [f"# Digest: {name} (draft, copy numbers into CLAIMS.md)", ""]
    lines += [*verdict_lines(d), ""]
    for model in sorted({r["model"] for r in rows}):
        lines += speedup_table(rows, model)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    text = digest(a.name)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
