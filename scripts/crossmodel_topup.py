"""Convergence top-up of the cross-model quick test (exploratory, D-028).

Reads the rows written by scripts/quick_topup.sh (results/raw/quick_models/speed.jsonl, time >= the
top-up start) for the four reliable models: 40 tuning items per language, 1 user, 128 tokens, vLLM
n-gram k=8 vs no speculation. Per model: speed-up per language (ratio of mean tokens/s) and the
Dravidian/English contrast (mean per-item speed-up on ta/te/kn/ml over English), with a 95%
bootstrap CI over items (items are parallel across languages and resampled jointly).

Usage: python scripts/crossmodel_topup.py   ->  results/tables/crossmodel_topup.csv
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
START = "2026-11-28T19:55"
LANGS = ["en", "hi", "ta", "te", "kn", "ml"]
TPW = {
    "qwen3-4b": "qwen3-4b",
    "navarasa-7b": "navarasa-7b",
    "gemma-4-e4b": "gemma-4-e4b",
    "sarvam-1": "sarvam-1",
}


def main() -> None:
    raw = REPO / "results/raw/quick_models"
    rows = [
        json.loads(x) for x in (raw / "speed.jsonl").read_text(encoding="utf-8").splitlines() if x
    ]
    rows = [r for r in rows if r["time"] >= START]
    tok = {}
    for x in (raw / "tokens.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            t = json.loads(x)
            tok[t["model"]] = t["tok_per_word"]
    g: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        g[(r["model"], r["arm"], r["lang"])][r["item"]] = r["output_tokens"] / r["latency_s"]
    out = []
    for m in dict.fromkeys(r["model"] for r in rows):
        ids = sorted(
            set.intersection(*(set(g[(m, a, x)]) for a in ("A0", "ngram8") for x in LANGS))
        )
        mat = np.array([[g[(m, "ngram8", x)][i] / g[(m, "A0", x)][i] for x in LANGS] for i in ids])
        rng = np.random.default_rng(1234)
        b = mat[rng.integers(0, len(mat), (2000, len(mat)))]
        boots = b[:, :, 2:].mean((1, 2)) / b[:, :, 0].mean(1)
        rec = {
            "model": m,
            "n_items": len(ids),
            "tokens_per_dravidian_word": float(np.mean([tok[TPW[m]][x] for x in LANGS[2:]])),
        }
        for x in LANGS:
            a = sum(g[(m, "ngram8", x)][i] for i in ids)
            rec[f"speedup_{x}"] = a / sum(g[(m, "A0", x)][i] for i in ids)
        rec["dravidian_over_english"] = float(mat[:, 2:].mean() / mat[:, 0].mean())
        rec["ci_lo"], rec["ci_hi"] = (float(q) for q in np.quantile(boots, [0.025, 0.975]))
        out.append(rec)
        print(
            f"{m:12} n={len(ids)} tok/word {rec['tokens_per_dravidian_word']:.2f}  "
            f"Dravidian/English {rec['dravidian_over_english']:.2f} "
            f"[{rec['ci_lo']:.2f}, {rec['ci_hi']:.2f}]"
        )
    dest = REPO / "results/tables/crossmodel_topup.csv"
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
