"""Smoke test per (engine, arm, model) (WS1 acceptance: `make smoke`).

Launches the server, sends one short prompt per language (greedy, bs=1), checks non-empty outputs
and, for speculative arms, non-zero speculative counters (/metrics delta), then shuts down. Writes
``results/raw/smoke/<run_id>/{requests.jsonl, run_manifest.json, server.log, telemetry.csv}``
and appends a line to ``results/raw/smoke/summary.jsonl``. Linux/WSL only.

Usage: python -m indicspec.harness.smoke --engine vllm --arm A0 --model qwen3-4b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from indicspec.env_probe import probe_gpu
from indicspec.harness import gpu_guard, promscrape
from indicspec.harness.loadgen import LoadSpec, Prompt, run_load, to_log_row
from indicspec.harness.manifest import build_manifest, now_utc
from indicspec.harness.schema import validate
from indicspec.harness.server import REPO, Server, build_plan, load_configs
from indicspec.harness.telemetry import TelemetrySampler, summarise
from indicspec.hub import local_snapshot as hub_local_snapshot

# Short instructions, one per language. Written for smoke testing only (not benchmark data).
SMOKE_TEXT = {
    "en": "Write two sentences about the monsoon.",
    "hi": "मानसून के बारे में दो वाक्य लिखिए।",
    "ta": "பருவமழை பற்றி இரண்டு வாக்கியங்கள் எழுதுங்கள்.",
    "te": "వర్షాకాలం గురించి రెండు వాక్యాలు రాయండి.",
    "kn": "ಮಳೆಗಾಲದ ಬಗ್ಗೆ ಎರಡು ವಾಕ್ಯಗಳನ್ನು ಬರೆಯಿರಿ.",
    "ml": "മഴക്കാലത്തെക്കുറിച്ച് രണ്ട് വാക്യങ്ങൾ എഴുതുക.",
}


def local_snapshot(repo: str, revision: str | None) -> str:
    return str(hub_local_snapshot(repo, revision))


def smoke_prompts(target_path: str, prompts_file: Path | None = None) -> list[Prompt]:
    """Default: one smoke instruction per language. With ``prompts_file`` (JSONL rows with
    prompt_id, language, task, messages) those prompts are used instead (config probes)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(target_path)
    items: list[tuple[str, str, str, list[dict[str, str]]]] = []
    if prompts_file is None:
        items = [
            (f"smoke-{lang}", lang, "smoke", [{"role": "user", "content": text}])
            for lang, text in SMOKE_TEXT.items()
        ]
    else:
        for line in prompts_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                items.append((r["prompt_id"], r["language"], r["task"], r["messages"]))
    out = []
    for pid, lang, task, messages in items:
        ids: Any = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
        )
        if not isinstance(ids, list):  # newer transformers may return a BatchEncoding
            ids = ids["input_ids"]
        token_ids: list[int] = [int(i) for i in ids]
        out.append(Prompt(pid, lang, task, token_ids))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["vllm", "sglang"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--prompts-file", type=Path, default=None)
    ap.add_argument("--spec-override", default=None, help="JSON merged into vLLM spec config")
    ap.add_argument("--tag", default="", help="suffix for the run id")
    ap.add_argument("--gpu-mem-util", type=float, default=None, help="default: max safe (D-015)")
    args = ap.parse_args(argv)

    engines, arms, models = load_configs()
    tgt = models["targets"][args.model]
    target_path = local_snapshot(tgt["repo"], tgt["revision"])
    arm_cfg = arms["arms"][args.arm]
    drafter_key = arm_cfg.get("drafter")
    drafter = None
    drafter_path = None
    if drafter_key:
        drafter_key = drafter_key.format(size=args.model.split("-")[-1])
        drafter = arms["drafters"][drafter_key]
        drafter_path = local_snapshot(drafter["repo"], drafter["revision"])

    if args.gpu_mem_util is None:
        headroom = float(engines[args.engine].get("mem_fraction_headroom", 0.01))
        args.gpu_mem_util = round(gpu_guard.max_safe_fraction(gpu_guard.query()) - headroom, 3)
    plan = build_plan(
        args.engine,
        args.arm,
        model=args.model,
        target_path=target_path,
        drafter_path=drafter_path,
        port=args.port,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem_util,
        overrides=json.loads(args.spec_override) if args.spec_override else None,
    )
    tag = f"-{args.tag}" if args.tag else ""
    run_id = f"smoke-{args.engine}-{args.arm}-{args.model}{tag}-{time.strftime('%Y%m%dT%H%M%S')}"
    out = REPO / "results/raw/smoke" / run_id
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        run_id=run_id,
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        engine=args.engine,
        engine_version=engines[args.engine]["version"],
        engine_launch_cmd=plan.cmd,
        target={"repo": tgt["repo"], "revision": tgt["revision"], "dtype": tgt["dtype"]},
        drafter=(
            {"repo": drafter["repo"], "revision": drafter["revision"], "method": args.arm}
            if drafter
            else None
        ),
        gpu=probe_gpu(),
        seeds={"engine": 1234, "request": 1234},
        start_utc=now_utc(),
    )
    prompts = smoke_prompts(target_path, args.prompts_file)
    spec = LoadSpec(
        f"http://127.0.0.1:{args.port}",
        args.model,
        max_tokens=args.max_tokens,
        extra_body=dict(engines[args.engine].get("request_extras") or {}),
    )
    metric_names = list(engines[args.engine].get("metrics", {}).values())
    status: dict[str, Any] = {
        "run_id": run_id,
        "engine": args.engine,
        "arm": args.arm,
        "model": args.model,
    }
    try:
        vram = gpu_guard.query()
        status["vram_before"] = {"total_mib": vram.total_mib, "used_mib": vram.used_mib}
        gpu_guard.preflight(args.gpu_mem_util, vram)
        with TelemetrySampler(out / "telemetry.csv"), Server(plan, out / "server.log") as srv:
            before = promscrape.scrape(srv.base_url, metric_names)
            results, wall = asyncio.run(run_load(spec, prompts))
            after = promscrape.scrape(srv.base_url, metric_names)
        rows = [
            to_log_row(
                r, spec, run_id=run_id, arm=args.arm, engine=args.engine, split="smoke", repeat=0
            )
            for r in results
        ]
        with open(out / "requests.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                validate(row, "request_log.v1")
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        d = promscrape.delta(before, after)
        tsum = summarise(out / "telemetry.csv", max_temp_c=200.0, min_sm_clock_ratio=0.0)
        vram_ok = gpu_guard.in_run_ok(tsum.peak_mem_mib, vram.total_mib)
        status.update(
            peak_vram_mib=tsum.peak_mem_mib,
            vram_ok=vram_ok,
            max_temp_c=tsum.max_temp_c,
            median_sm_clock=tsum.median_sm_clock,
        )
        errors = [r.error for r in results if r.error]
        empty = [r.prompt.prompt_id for r in results if not r.output_text.strip()]
        spec_ok = (args.arm == "A0") or any(v > 0 for v in d.values())
        status.update(
            ok=not errors and not empty and spec_ok and vram_ok,
            errors=errors,
            empty_outputs=empty,
            spec_counter_delta=d,
            wall_s=wall,
            outputs={r.prompt.language: r.output_text[:200] for r in results},
        )
    except Exception as exc:  # recorded, not hidden: goes to ENGINE_STATUS
        status.update(ok=False, exception=repr(exc)[:3000])
    manifest["end_utc"] = now_utc()
    manifest["telemetry_csv"] = str((out / "telemetry.csv").relative_to(REPO))
    validate(manifest, "run_manifest.v1")
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(REPO / "results/raw/smoke/summary.jsonl", "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {k: status.get(k) for k in ("run_id", "ok", "exception", "spec_counter_delta")}
            )
            + "\n"
        )
    print(json.dumps(status, indent=1, ensure_ascii=False)[:3000])
    return 0 if status.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
