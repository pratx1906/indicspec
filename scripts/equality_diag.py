"""Exploratory diagnostic for the greedy-equality gate failure (2026-11-26). WSL/GPU.

Two checks on the plain target (A0) only, per model:
  1. Rank check: for every flagged drafter divergence in results/raw/equality_<model>/<arm>.jsonl,
     re-run A0 at batch size 1 with top-5 log-probs; report the rank of the drafter token at the
     divergence position (a near-tie flip predicts rank 2).
  2. Batch control: send the same 120 prompts to A0 with 16 requests in flight and compare with the
     batch-1 outputs. If divergences appear with the same top-2 gap profile, batch-shape numerics in
     the target alone explain the gate failures (no drafter involved).

Output: results/raw/equality_<model>/diag.json
Usage: python scripts/equality_diag.py --model qwen3-4b --arms A1 A5
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from indicspec.harness import gpu_guard
from indicspec.harness.equality import classify, compare
from indicspec.harness.equality_gate import NEAR_TIE, top2_gap
from indicspec.harness.server import Server, build_plan, load_configs
from indicspec.harness.sweep import load_prompts
from indicspec.hub import local_snapshot

REPO = Path(__file__).resolve().parents[1]
LANGS = ("en", "hi", "ta", "te", "kn", "ml")


def _post(
    client: httpx.Client, url: str, model: str, ids: list[int]
) -> tuple[list[int], list[Any]]:
    body = {
        "model": model,
        "prompt": ids,
        "max_tokens": 256,
        "temperature": 0.0,
        "seed": 1234,
        "return_token_ids": True,
        "logprobs": 5,
        "return_tokens_as_token_ids": True,
    }
    r = client.post(f"{url}/v1/completions", json=body)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    return [int(t) for t in ch["token_ids"]], list(
        (ch.get("logprobs") or {}).get("top_logprobs") or []
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--inflight", type=int, default=16)
    a = ap.parse_args()
    out_dir = REPO / "results/raw" / f"equality_{a.model}"
    engines, _, models = load_configs()
    tgt = models["targets"][a.model]
    vram = gpu_guard.query()
    frac = round(
        gpu_guard.max_safe_fraction(vram)
        - float(engines["vllm"].get("mem_fraction_headroom", 0.01)),
        3,
    )
    gpu_guard.preflight(frac, vram)
    gpu_guard.power_preflight(gpu_guard.power_state(), 140.0)
    prompts, _ = load_prompts(
        REPO / "data/bench",
        "report",
        LANGS,
        ("chat", "qa", "summarisation", "translation"),
        4096 - 256,
        20,
        1234,
    )
    items = [
        (f"{lang}|{p.prompt_id}", lang, p.token_ids) for lang, ps in prompts.items() for p in ps
    ]
    plan = build_plan(
        "vllm",
        "A0",
        model=a.model,
        target_path=str(local_snapshot(tgt["repo"], tgt["revision"])),
        drafter_path=None,
        gpu_mem_util=frac,
        seed=1234,
    )
    with Server(plan, out_dir / "server-diag.log") as srv, httpx.Client(timeout=900) as client:
        bs1 = {k: _post(client, srv.base_url, a.model, ids) for k, _, ids in items}
        with ThreadPoolExecutor(a.inflight) as ex:
            futs = {k: ex.submit(_post, client, srv.base_url, a.model, ids) for k, _, ids in items}
            batched = {k: f.result()[0] for k, f in futs.items()}

    # 1. rank of the drafter's token among the target's top-5 at each flagged divergence
    ranks: dict[str, Counter[str]] = {}
    for arm in a.arms:
        c: Counter[str] = Counter()
        for line in (out_dir / f"{arm}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["status"] not in ("suspected_bug", "near_tie") or r["arm_token"] is None:
                continue
            ids, top = bs1[r["prompt_id"]]
            if r["position"] >= len(top) or ids[r["position"]] != r["baseline_token"]:
                c["baseline_not_reproduced"] += 1
                continue
            order = sorted(top[r["position"]].items(), key=lambda kv: -kv[1])
            toks = [int(k.split(":")[1]) for k, _ in order]
            c[
                f"rank{toks.index(r['arm_token']) + 1}"
                if r["arm_token"] in toks
                else "outside_top5"
            ] += 1
        ranks[arm] = c

    # 2. batch control: plain target, 16 in flight vs batch size 1
    ctrl: Counter[str] = Counter()
    gaps: list[float] = []
    for k, _, _ in items:
        ids, top = bs1[k]
        d = compare(k, ids, batched[k])
        if d is None:
            ctrl["equal"] += 1
            continue
        g = top2_gap(top[d.position] if d.position < len(top) else None)
        ctrl[classify(g, NEAR_TIE)] += 1
        if not math.isinf(g):
            gaps.append(round(g, 3))
    res = {
        "model": a.model,
        "rank_of_drafter_token_at_divergence": {k: dict(v) for k, v in ranks.items()},
        "a0_batch_control": {"inflight": a.inflight, "counts": dict(ctrl), "gaps": sorted(gaps)},
    }
    (out_dir / "diag.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
