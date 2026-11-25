"""Coverage-only upper bound on acceptance length for pruned-vocabulary drafters (H1, exploratory).

Assume an *oracle* drafter that always proposes the correct next token whenever that token is in
its draft vocabulary, and that drafts ``k`` tokens per step (chain drafting). At each verification
step starting at position i it gets ``a = min(k, run)`` accepted tokens, where ``run`` is the number
of consecutive in-vocabulary tokens starting at i, plus one bonus token from the target:

    emitted = a + 1;  i += emitted;  tau_oracle = mean(emitted)

No real drafter can beat this on the same text, so it isolates the ceiling imposed by vocabulary
coverage alone. A full-vocabulary oracle gets ``k + 1`` everywhere (except at the sequence end).
Computed on reference text (proxy for target generations; exploratory).

Usage: python -m indicspec.analysis.oracle_bound --drafter A3=DIR --drafter A4=DIR [--k 3 5 7]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from indicspec.analysis.stats import bootstrap_ci
from indicspec.analysis.tokentax import LANGS, read_bench, unique_refs
from indicspec.tokens.vocab import PinnedTokenizer, load_tokenizer

REPO = Path(__file__).resolve().parents[3]


def oracle_steps(in_vocab: Sequence[bool], k: int) -> list[int]:
    """Emitted tokens per verification step for the oracle chain drafter."""
    n = len(in_vocab)
    out = []
    i = 0
    while i < n:
        run = 0
        while run < k and i + run < n and in_vocab[i + run]:
            run += 1
        emitted = run + 1  # accepted drafts + bonus token
        out.append(min(emitted, n - i))
        i += emitted
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=REPO / "data/bench")
    ap.add_argument("--drafter", action="append", required=True, help="NAME=checkpoint_dir")
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--out", type=Path, default=REPO / "results/tables/oracle_tau_bound.csv")
    args = ap.parse_args(argv)
    from indicspec.tokens.draftvocab import load_draft_vocab

    cfg = yaml.safe_load((REPO / "configs/models.yaml").read_text(encoding="utf-8"))
    q = cfg["targets"]["qwen3-8b"]
    tok = load_tokenizer(PinnedTokenizer(q["repo"], q["revision"]))
    refs = unique_refs(read_bench(args.bench))
    seqs: dict[str, list[list[int]]] = defaultdict(list)
    for (_task, lang, _iid), text in sorted(refs.items()):
        seqs[lang].append(tok.encode(text, add_special_tokens=False).ids)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "drafter",
                "k",
                "lang",
                "n_seqs",
                "tau_oracle",
                "ci95_lo",
                "ci95_hi",
                "full_vocab_tau_oracle",
            ]
        )
        for spec in args.drafter:
            name, path = spec.split("=", 1)
            dv = load_draft_vocab(Path(path)).target_ids
            for k in args.k:
                for lang in LANGS:
                    per_seq = []
                    full = []
                    for ids in seqs.get(lang, []):
                        if not ids:
                            continue
                        st = oracle_steps([i in dv for i in ids], k)
                        per_seq.append(float(np.mean(st)))
                        full.append(float(np.mean(oracle_steps([True] * len(ids), k))))
                    if not per_seq:
                        continue
                    ci = bootstrap_ci(per_seq, n_boot=2000)
                    w.writerow(
                        [
                            name,
                            k,
                            lang,
                            len(per_seq),
                            f"{ci.estimate:.3f}",
                            f"{ci.lo:.3f}",
                            f"{ci.hi:.3f}",
                            f"{np.mean(full):.3f}",
                        ]
                    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
