"""Greedy-equality correctness gate (docs/PREREG.md §5; brief §9). Linux/WSL only for the CLI.

For each arm, launch its server, send 20 report-split prompts per language one at a time (batch size
1, greedy, max_tokens 256) and collect output token ids. A0 additionally returns the top-5 log-probs
per generated position. Each arm's ids are compared with A0's: at the first divergence, the target's
top-2 logit gap (difference of the two largest log-probs; equal to the logit gap because both share
the softmax normaliser) and the rank of the arm's token among the target's candidates classify the
mismatch.

Rule (PREREG Amendment 2, D-021): near-tie if the arm's token is among the target's top-3 and the
gap is <= 1.0 (8 bf16 logit steps); otherwise a suspected bug, which blocks the arm until explained.
The frozen rule (gap < 0.1, below one bf16 step) is still recorded as ``frozen_status``.

vLLM 0.34.0 fields used (verified in the installed package):
* request ``logprobs`` (int) and ``return_tokens_as_token_ids`` (bool),
  ``entrypoints/openai/completion/protocol.py`` lines 63 and 164; ``return_token_ids`` line 172;
* response ``choices[].token_ids`` (protocol.py:688) and ``choices[].logprobs.top_logprobs``, a list
  of ``{token: logprob}`` dicts (protocol.py:672) whose keys are ``"token_id:{id}"`` when
  ``return_tokens_as_token_ids`` is set (``entrypoints/generate/base/serving.py:299-300``) and which
  hold the first ``logprobs + 1`` candidates (serving.py:345-353).

Usage (WSL):
  python -m indicspec.harness.equality_gate --engine vllm --model qwen3-8b --arms A3 A4 A5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from indicspec.harness.equality import classify, compare

NEAR_TIE = 0.1  # frozen rule (PREREG §5); recorded, no longer binding
NEAR_TIE_GAP = 1.0  # PREREG Amendment 2
NEAR_TIE_TOPK = 3
REPO = Path(__file__).resolve().parents[3]


def top2_gap(top_logprobs_at_pos: Mapping[str, float] | None) -> float:
    """Gap between the two largest log-probs at one position (inf if fewer than two)."""
    if not top_logprobs_at_pos:
        return math.inf
    vals = sorted((float(v) for v in top_logprobs_at_pos.values()), reverse=True)
    return vals[0] - vals[1] if len(vals) >= 2 else math.inf


def token_rank(top_logprobs_at_pos: Mapping[str, float] | None, token: int | None) -> int | None:
    """1-based rank of ``token`` among candidates keyed ``"token_id:<id>"`` (None if absent)."""
    if not top_logprobs_at_pos or token is None:
        return None
    order = sorted(top_logprobs_at_pos.items(), key=lambda kv: -float(kv[1]))
    ids = [int(k.rsplit(":", 1)[1]) for k, _ in order]
    return ids.index(token) + 1 if token in ids else None


def amended_status(gap: float, rank: int | None) -> str:
    """PREREG Amendment 2: near-tie iff the arm token is in the target's top-3 and gap <= 1.0."""
    if rank is not None and rank <= NEAR_TIE_TOPK and gap <= NEAR_TIE_GAP:
        return "near_tie"
    return "suspected_bug"


def compare_arm(
    baseline: Mapping[str, tuple[list[int], list[dict[str, float] | None]]],
    arm_ids: Mapping[str, list[int]],
    languages: Mapping[str, str],
) -> list[dict[str, Any]]:
    """One row per prompt: equal / near_tie / suspected_bug, with divergence details."""
    rows: list[dict[str, Any]] = []
    for pid, (base_ids, base_top) in sorted(baseline.items()):
        if pid not in arm_ids:
            rows.append({"prompt_id": pid, "language": languages.get(pid), "status": "missing"})
            continue
        d = compare(pid, base_ids, arm_ids[pid])
        if d is None:
            rows.append({"prompt_id": pid, "language": languages.get(pid), "status": "equal"})
            continue
        top = base_top[d.position] if d.position < len(base_top) else None
        gap = top2_gap(top)
        rank = token_rank(top, d.arm_token)
        rows.append(
            {
                "prompt_id": pid,
                "language": languages.get(pid),
                "status": amended_status(gap, rank),
                "frozen_status": classify(gap, NEAR_TIE),
                "arm_token_rank": rank,
                "position": d.position,
                "baseline_token": d.baseline_token,
                "arm_token": d.arm_token,
                "baseline_len": d.baseline_len,
                "arm_len": d.arm_len,
                "top2_logit_gap": None if math.isinf(gap) else gap,
            }
        )
    return rows


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_lang: dict[str, Counter[str]] = {}
    for r in rows:
        by_lang.setdefault(str(r.get("language")), Counter())[str(r["status"])] += 1
    total = Counter(str(r["status"]) for r in rows)
    return {
        "n": len(rows),
        "counts": dict(total),
        "by_language": {k: dict(v) for k, v in sorted(by_lang.items())},
        "pass": total.get("suspected_bug", 0) == 0 and total.get("missing", 0) == 0,
    }


def parse_completion(resp: Mapping[str, Any]) -> tuple[list[int], list[dict[str, float] | None]]:
    ch = resp["choices"][0]
    ids = [int(t) for t in (ch.get("token_ids") or [])]
    lp = ch.get("logprobs") or {}
    return ids, list(lp.get("top_logprobs") or [])


def _run_arm(engine: str, model: str, arm: str, n_per_lang: int, seed: int) -> dict[str, Any]:
    """Launch one arm's server and collect greedy outputs (GPU; WSL/Linux)."""
    import httpx

    from indicspec.harness import gpu_guard
    from indicspec.harness.server import Server, build_plan, load_configs
    from indicspec.harness.sweep import load_prompts
    from indicspec.hub import local_snapshot

    engines, arms, models = load_configs()
    tgt = models["targets"][model]
    target_path = str(local_snapshot(tgt["repo"], tgt["revision"]))
    drafter_path = None
    dkey = arms["arms"][arm].get("drafter")
    if dkey:
        d = arms["drafters"][dkey.format(size=model.split("-")[-1])]
        drafter_path = str(local_snapshot(d["repo"], d["revision"]))
    vram = gpu_guard.query()
    frac = round(
        gpu_guard.max_safe_fraction(vram)
        - float(engines[engine].get("mem_fraction_headroom", 0.01)),
        3,
    )
    gpu_guard.preflight(frac, vram)
    gpu_guard.power_preflight(gpu_guard.power_state(), 140.0)
    prompts, _ = load_prompts(
        REPO / "data/bench",
        "report",
        ("en", "hi", "ta", "te", "kn", "ml"),
        ("chat", "qa", "summarisation", "translation"),
        4096 - 256,
        n_per_lang,
        seed,
    )
    plan = build_plan(
        engine,
        arm,
        model=model,
        target_path=target_path,
        drafter_path=drafter_path,
        gpu_mem_util=frac,
        seed=seed,
    )
    out: dict[str, Any] = {"ids": {}, "top": {}, "lang": {}}
    stamp = time.strftime("%Y%m%dT%H%M%S")
    log = REPO / "results/raw" / f"equality_{model}" / f"server-{arm}-{stamp}.log"
    with Server(plan, log) as srv, httpx.Client(timeout=600) as client:
        for lang, ps in prompts.items():
            for p in ps:
                body: dict[str, Any] = {
                    "model": model,
                    "prompt": p.token_ids,
                    "max_tokens": 256,
                    "temperature": 0.0,
                    "seed": seed,
                    "return_token_ids": True,
                }
                if arm == "A0":
                    body.update(logprobs=5, return_tokens_as_token_ids=True)
                r = client.post(f"{srv.base_url}/v1/completions", json=body)
                r.raise_for_status()
                ids, top = parse_completion(r.json())
                key = f"{lang}|{p.prompt_id}"
                out["ids"][key], out["top"][key], out["lang"][key] = ids, top, lang
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="vllm", choices=["vllm", "sglang"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--n-per-lang", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args(argv)
    out_dir = REPO / "results/raw" / f"equality_{a.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _run_arm(a.engine, a.model, "A0", a.n_per_lang, a.seed)
    baseline = {k: (base["ids"][k], base["top"][k]) for k in base["ids"]}
    summary: dict[str, Any] = {
        "engine": a.engine,
        "model": a.model,
        "rule": f"PREREG Amendment 2: top-{NEAR_TIE_TOPK} and gap <= {NEAR_TIE_GAP}",
        "frozen_near_tie": NEAR_TIE,
    }
    for arm in a.arms:
        res = _run_arm(a.engine, a.model, arm, a.n_per_lang, a.seed)
        rows = compare_arm(baseline, res["ids"], base["lang"])
        with open(out_dir / f"{arm}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        summary[arm] = summarise(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0 if all(summary[arm]["pass"] for arm in a.arms) else 1


if __name__ == "__main__":
    raise SystemExit(main())
