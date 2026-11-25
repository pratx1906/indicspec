"""Graft pre-check, part 2: can a linear map predict draft lm_head rows from target rows? (D-010)

Direct row copying is unprincipled when draft and target rows are not aligned (graft_check.py).
A training-free alternative: fit a ridge-regularised linear map ``W`` (hidden x hidden) such that
``target_row @ W ~= draft_row`` on the tokens both vocabularies share, then use ``W`` to synthesise
draft rows for missing tokens. Fit on 90% of shared tokens, report cosine on the held-out 10%
(vs the identity map and a shuffled control). A high held-out cosine makes grafting principled;
a low one rules it out. Exploratory (PLAN P3.5).

Usage: python -m indicspec.analysis.graft_linmap --draft REPO --draft-revision REV --out PATH
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


def _rowcos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=1)
    return num / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def _q(x: np.ndarray) -> dict[str, float]:
    p = np.quantile(x, [0.05, 0.5, 0.95])
    return {"mean": float(x.mean()), "p05": float(p[0]), "median": float(p[1]), "p95": float(p[2])}


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge: W = (X^T X + lam I)^-1 X^T Y (float64)."""
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    a = x64.T @ x64
    a[np.diag_indices_from(a)] += lam
    return np.linalg.solve(a, x64.T @ y64)


def run(
    draft_repo: str, draft_rev: str | None, target_key: str, seed: int = 1234
) -> dict[str, Any]:
    cfg = yaml.safe_load((REPO / "configs/models.yaml").read_text(encoding="utf-8"))
    tgt = cfg["targets"][target_key]
    ddir = local_snapshot(draft_repo, draft_rev)
    tdir = local_snapshot(tgt["repo"], tgt["revision"])
    dfiles, tfiles = sorted(ddir.glob("*.safetensors")), sorted(tdir.glob("*.safetensors"))
    ids = load_draft_vocab(ddir).draft_to_target
    d_names = {k for f in dfiles for k in list_tensors(f)}
    d_lm_name = next(k for k in d_names if k.endswith("lm_head.weight"))
    y = load_tensor(find_tensor(dfiles, d_lm_name)[0], d_lm_name)
    t_lm = load_tensor(find_tensor(tfiles, "lm_head.weight")[0], "lm_head.weight")
    x = t_lm[ids]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_test = len(ids) // 10
    te, tr = perm[:n_test], perm[n_test:]
    res: dict[str, Any] = {
        "schema": "indicspec.graft_linmap/1",
        "draft": {"repo": draft_repo, "revision": draft_rev},
        "target": {"repo": tgt["repo"], "revision": tgt["revision"]},
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "identity_heldout_cos": _q(_rowcos(x[te], y[te])),
        "ridge": {},
    }
    scale = float(np.mean(np.sum(x[tr].astype(np.float64) ** 2, axis=1)))
    for lam_rel in (1e-3, 1e-1, 10.0):
        w = fit_ridge(x[tr], y[tr], lam_rel * scale)
        pred = (x[te].astype(np.float64) @ w).astype(np.float32)
        ctrl = (x[rng.permutation(te)].astype(np.float64) @ w).astype(np.float32)
        mu_y = y[tr].mean(axis=0, keepdims=True)
        pc, yc = pred - mu_y, y[te] - mu_y
        # retrieval: is the predicted row closest (cosine, centred) to its own true row
        # among all held-out rows?  chance = 1 / n_test
        pn = pc / np.maximum(np.linalg.norm(pc, axis=1, keepdims=True), 1e-12)
        yn = yc / np.maximum(np.linalg.norm(yc, axis=1, keepdims=True), 1e-12)
        sim = pn @ yn.T
        top1 = float(np.mean(sim.argmax(axis=1) == np.arange(len(te))))
        rank = (sim > sim[np.arange(len(te)), np.arange(len(te))][:, None]).sum(axis=1)
        res["ridge"][str(lam_rel)] = {
            "centred_heldout_cos": _q(_rowcos(pc, yc)),
            "centred_control_cos": _q(_rowcos(ctrl - mu_y, yc)),
            "retrieval_top1": top1,
            "retrieval_top10": float(np.mean(rank < 10)),
            "retrieval_chance_top1": 1.0 / len(te),
            "heldout_cos": _q(_rowcos(pred, y[te])),
            "shuffled_control_cos": _q(_rowcos(ctrl, y[te])),
            "train_cos_median": float(
                np.median(_rowcos((x[tr].astype(np.float64) @ w).astype(np.float32), y[tr]))
            ),
        }
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--draft-revision", default=None)
    ap.add_argument("--target", default="qwen3-8b")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = run(a.draft, a.draft_revision, a.target)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "identity": res["identity_heldout_cos"]["median"],
                **{
                    k: (
                        v["heldout_cos"]["median"],
                        v["shuffled_control_cos"]["median"],
                        v["train_cos_median"],
                    )
                    for k, v in res["ridge"].items()
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
