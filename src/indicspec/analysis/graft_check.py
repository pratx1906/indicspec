"""Graft pre-check (PLAN P0, D-010): does a pruned EAGLE-3 head's lm_head match the target's?

Vocabulary grafting (N1) copies target lm_head rows into the draft head for tokens missing from
the draft vocabulary. That is only principled if the draft's existing rows are close to the
target's rows for the same tokens (e.g. the head was initialised from the target and stayed
close). This script measures, for every draft id d with target id t = d2t-mapped id:

* cosine(draft_lm_head[d], target_lm_head[t])  -- matched pairs
* the same with a random permutation of target ids -- null distribution
* relative norm ||draft row|| / ||target row||
* comparison of the final RMSNorm weights (draft ``norm.weight`` vs target ``model.norm.weight``)

Usage (inside WSL, models in the HF cache):
  python -m indicspec.analysis.graft_check --draft RedHatAI/Qwen3-8B-speculator.eagle3 \
      --out results/tables/graft_alignment_A3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from indicspec.hub import local_snapshot
from indicspec.tokens.draftvocab import load_draft_vocab
from indicspec.tokens.safetensors_np import find_tensor, list_tensors, load_tensor

REPO = Path(__file__).resolve().parents[3]


def _snapshot(repo: str, revision: str | None) -> Path:
    return local_snapshot(repo, revision)


def _summary(x: np.ndarray) -> dict[str, float]:
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "mean": float(x.mean()),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
    }


def _rowcos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-12)


def run(
    draft_repo: str, draft_rev: str | None, target_key: str, seed: int = 1234
) -> dict[str, Any]:
    cfg = yaml.safe_load((REPO / "configs/models.yaml").read_text(encoding="utf-8"))
    tgt = cfg["targets"][target_key]
    ddir = _snapshot(draft_repo, draft_rev)
    tdir = _snapshot(tgt["repo"], tgt["revision"])
    dfiles = sorted(ddir.glob("*.safetensors"))
    tfiles = sorted(tdir.glob("*.safetensors"))

    dv = load_draft_vocab(ddir)
    d_tensors = {k: v for f in dfiles for k, v in list_tensors(f).items()}
    d_lm_name = next(k for k in d_tensors if k.endswith("lm_head.weight"))
    d_lm = load_tensor(find_tensor(dfiles, d_lm_name)[0], d_lm_name)

    t_lm_name = "lm_head.weight"
    try:
        tf, _, _ = find_tensor(tfiles, t_lm_name)
    except KeyError:  # tied embeddings
        t_lm_name = "model.embed_tokens.weight"
        tf, _, _ = find_tensor(tfiles, t_lm_name)
    t_lm = load_tensor(tf, t_lm_name)

    ids = dv.draft_to_target
    if d_lm.shape != (len(ids), t_lm.shape[1]):
        raise ValueError(
            f"draft lm_head {d_lm.shape} vs draft vocab {len(ids)} / hidden {t_lm.shape[1]}"
        )
    matched = t_lm[ids]
    cos = _rowcos(d_lm, matched)
    rng = np.random.default_rng(seed)
    null = _rowcos(d_lm, t_lm[rng.permutation(ids)])
    rel_norm = np.linalg.norm(d_lm, axis=1) / np.maximum(np.linalg.norm(matched, axis=1), 1e-12)
    max_abs_diff = float(np.abs(d_lm - matched).max())

    norms: dict[str, Any] = {}
    d_norm_name = next((k for k in d_tensors if k in ("norm.weight", "model.norm.weight")), None)
    if d_norm_name:
        dn = load_tensor(find_tensor(dfiles, d_norm_name)[0], d_norm_name)
        tn = load_tensor(find_tensor(tfiles, "model.norm.weight")[0], "model.norm.weight")
        norms = {
            "draft_norm_tensor": d_norm_name,
            "cosine": float(_rowcos(dn[None], tn[None])[0]),
            "max_abs_diff": float(np.abs(dn - tn).max()),
            "draft_mean": float(dn.mean()),
            "target_mean": float(tn.mean()),
        }

    return {
        "schema": "indicspec.graft_check/1",
        "draft": {"repo": draft_repo, "revision": draft_rev, "path": str(ddir)},
        "target": {"repo": tgt["repo"], "revision": tgt["revision"], "lm_head_tensor": t_lm_name},
        "draft_vocab_size": len(ids),
        "target_vocab_size": int(t_lm.shape[0]),
        "hidden": int(t_lm.shape[1]),
        "draft_tensors": {k: v for k, v in sorted(d_tensors.items())},
        "cosine_matched": _summary(cos),
        "cosine_null_permuted": _summary(null),
        "relative_row_norm": _summary(rel_norm),
        "max_abs_diff_matched": max_abs_diff,
        "identical_rows_fraction": float(np.mean(np.all(d_lm == matched, axis=1))),
        "final_norm": norms,
        "seed": seed,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--draft-revision", default=None)
    ap.add_argument("--target", default="qwen3-8b")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    res = run(args.draft, args.draft_revision, args.target)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("cosine_matched", "cosine_null_permuted")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
