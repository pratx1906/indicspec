"""Token tax, fragmentation and draft-vocab coverage on reference text (WS2.5, WS3.2, WS3.3).

Reads ``data/bench/*.jsonl`` (fields used: item_id, task, lang, reference_text), tokenises each
reference text with the pinned tokenizer(s) (no special tokens) and writes:

* ``results/tables/token_tax.csv``      per tokenizer x task x lang: tokens/char, tokens/word, and
  parity vs English on the *same item ids* (ratio of sums + per-item median, bootstrap 95% CI).
* ``results/tables/fragmentation_ref.csv``  per task x lang: share of reference tokens per class.
* ``results/tables/coverage_ref.csv``   per drafter x task x lang x class: frequency-weighted
  coverage.

Coverage here is on *reference* text, a proxy for target-generated text (the spec's definition);
it is labelled exploratory until target generations exist (WS3.3).

Usage: python -m indicspec.analysis.tokentax [--drafter NAME=HF_REPO@REV ...]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from indicspec.analysis.stats import bootstrap_ci
from indicspec.tokens.classes import TokenClass
from indicspec.tokens.vocab import (
    PinnedTokenizer,
    classify_vocab,
    coverage,
    fetch_tokenizer_json,
    load_tokenizer,
)

REPO = Path(__file__).resolve().parents[3]
LANGS = ["en", "hi", "ta", "te", "kn", "ml"]


def read_bench(bench_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in sorted(bench_dir.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def unique_refs(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    """(task, lang, item_id) -> reference_text, de-duplicated (several prompts may share a text)."""
    out: dict[tuple[str, str, str], str] = {}
    for r in rows:
        out.setdefault((r["task"], r["lang"], str(r["item_id"])), r["reference_text"])
    return out


def parity(
    counts: dict[str, int], en_counts: dict[str, int], seed: int = 1234
) -> tuple[float, float, float, float, int]:
    """Parity of a language vs English on shared items.

    Returns (ratio_of_sums, median_item_ratio, ci_lo, ci_hi, n_items); CI is a percentile
    bootstrap over items of the ratio of sums.
    """
    shared = sorted(set(counts) & set(en_counts))
    if not shared:
        nan = float("nan")
        return nan, nan, nan, nan, 0
    a = np.array([counts[i] for i in shared], dtype=float)
    b = np.array([en_counts[i] for i in shared], dtype=float)
    keep = b > 0
    a, b = a[keep], b[keep]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(2000, a.size))
    boots = a[idx].sum(axis=1) / b[idx].sum(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    ratio = float(a.sum()) / float(b.sum())
    return ratio, float(np.median(a / b)), float(lo), float(hi), int(a.size)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=REPO / "data/bench")
    ap.add_argument("--out", type=Path, default=REPO / "results/tables")
    ap.add_argument(
        "--drafter",
        action="append",
        default=[],
        help="NAME=local_checkpoint_dir (EAGLE-3 heads with d2t/t2d)",
    )
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((REPO / "configs/models.yaml").read_text(encoding="utf-8"))
    refs = unique_refs(read_bench(args.bench))
    if not refs:
        raise SystemExit(f"no rows in {args.bench}")
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizers = {"qwen3": cfg["targets"]["qwen3-8b"], **cfg.get("compare_tokenizers", {})}
    qwen3_ids: dict[tuple[str, str, str], list[int]] = {}

    with open(args.out / "token_tax.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "tokenizer",
                "revision",
                "task",
                "lang",
                "n_items",
                "chars",
                "words",
                "tokens",
                "tokens_per_char",
                "tokens_per_word",
                "parity_vs_en_ratio_of_sums",
                "parity_vs_en_median_item",
                "parity_ci95_lo",
                "parity_ci95_hi",
                "parity_n_items",
            ]
        )
        for tname, t in tokenizers.items():
            tok = load_tokenizer(PinnedTokenizer(t["repo"], t["revision"]))
            per: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
            agg: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
            for (task, lang, iid), text in sorted(refs.items()):
                ids = tok.encode(text, add_special_tokens=False).ids
                if tname == "qwen3":
                    qwen3_ids[(task, lang, iid)] = ids
                per[(task, lang)][iid] = len(ids)
                agg[(task, lang)].update(
                    {"chars": len(text), "words": len(text.split()), "tokens": len(ids)}
                )
            for task in sorted({k[0] for k in per}):
                for lang in LANGS:
                    if (task, lang) not in per:
                        continue
                    c = agg[(task, lang)]
                    p = parity(per[(task, lang)], per.get((task, "en"), {}))
                    w.writerow(
                        [
                            tname,
                            t["revision"],
                            task,
                            lang,
                            len(per[(task, lang)]),
                            c["chars"],
                            c["words"],
                            c["tokens"],
                            f"{c['tokens'] / max(c['chars'], 1):.4f}",
                            f"{c['tokens'] / max(c['words'], 1):.4f}",
                            *(f"{x:.4f}" for x in p[:4]),
                            p[4],
                        ]
                    )

    q = cfg["targets"]["qwen3-8b"]
    info = classify_vocab(fetch_tokenizer_json(PinnedTokenizer(q["repo"], q["revision"])))
    classes = [str(c) for c in TokenClass]
    with open(args.out / "fragmentation_ref.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "lang", "n_tokens", *[f"share_{c}" for c in classes]])
        by: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for (task, lang, _), ids in qwen3_ids.items():
            by[(task, lang)].update(str(info[i].cls) for i in ids)
            by[("ALL", lang)].update(str(info[i].cls) for i in ids)
        for (task, lang), cnt in sorted(by.items()):
            n = sum(cnt.values())
            w.writerow([task, lang, n, *[f"{cnt[c] / n:.4f}" for c in classes]])

    if args.drafter:
        from indicspec.tokens.draftvocab import load_draft_vocab

        with open(args.out / "coverage_ref.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "drafter",
                    "task",
                    "lang",
                    "class",
                    "weighted_coverage",
                    "unique_coverage",
                    "n_tokens",
                    "n_unique",
                    "ci95_lo",
                    "ci95_hi",
                ]
            )
            for spec in args.drafter:
                name, path = spec.split("=", 1)
                dv = load_draft_vocab(Path(path)).target_ids
                groups: dict[tuple[str, str, str], list[list[int]]] = defaultdict(list)
                for (task, lang, _), ids in qwen3_ids.items():
                    for t in (task, "ALL"):
                        groups[(t, lang, "ALL")].append(ids)
                        for cname in classes:
                            sub = [i for i in ids if str(info[i].cls) == cname]
                            if sub:
                                groups[(t, lang, cname)].append(sub)
                for (task, lang, cname), seqs in sorted(groups.items()):
                    flat = [i for s in seqs for i in s]
                    wc, uc, n, nu = coverage(flat, dv)
                    per_item = [np.mean([i in dv for i in s]) for s in seqs]
                    ci = bootstrap_ci(per_item, n_boot=1000) if len(per_item) > 1 else None
                    w.writerow(
                        [
                            name,
                            task,
                            lang,
                            cname,
                            f"{wc:.4f}",
                            f"{uc:.4f}",
                            n,
                            nu,
                            f"{ci.lo:.4f}" if ci else "",
                            f"{ci.hi:.4f}" if ci else "",
                        ]
                    )
    print(f"wrote tables to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
