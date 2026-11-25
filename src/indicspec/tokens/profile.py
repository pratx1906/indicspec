"""Vocabulary class profile (WS3.1/3.2): class x script counts of the target vocabulary.

Usage: python -m indicspec.tokens.profile --model qwen3-8b --out results/tables/vocab_classes.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import yaml

from indicspec.tokens.vocab import PinnedTokenizer, classify_vocab, fetch_tokenizer_json

REPO = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--out", type=Path, default=REPO / "results/tables/vocab_classes.csv")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((REPO / "configs/models.yaml").read_text(encoding="utf-8"))
    t = cfg["targets"][args.model]
    info = classify_vocab(fetch_tokenizer_json(PinnedTokenizer(t["repo"], t["revision"])))
    counts = Counter((str(v.cls), v.script or "-") for v in info.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# source", f"{t['repo']}@{t['revision']}", "tokenizer.json"])
        w.writerow(["class", "script", "n_tokens"])
        for (cls, script), n in sorted(counts.items()):
            w.writerow([cls, script, n])
    print(f"wrote {args.out} ({len(info)} vocab entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
