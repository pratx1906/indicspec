"""Load generator for OpenAI-compatible completion servers (vLLM, SGLang) (WS4).

* Prompts are sent **pre-tokenised** (``prompt`` = list of token ids, as SPEED-Bench does, so both
  engines see identical inputs; spec §2.4 / WS2.5).
* ``stream=True`` with ``stream_options.include_usage`` so the final chunk carries the exact
  ``completion_tokens`` count. With speculative decoding one streamed chunk can carry several
  tokens, so per-chunk gaps are *not* per-token latencies; TPOT is computed from end-to-end time
  and the token count (spec §8.3), and chunk gaps are logged only as raw data.
* Closed loop: ``concurrency`` workers, each sends its next request when the previous one finishes.
* Open loop: Poisson arrivals at ``request_rate`` req/s (seeded exponential gaps), concurrency cap.

Each finished request becomes one row matching ``schemas/request_log.v1.schema.json``.
The request-field names follow the OpenAI completions API; engine-specific extras are passed
through ``extra_body`` and must be checked against the installed engine version.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from indicspec.analysis.metrics import tpot as tpot_fn


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    language: str
    task: str
    token_ids: list[int]


@dataclass
class LoadSpec:
    base_url: str
    model: str
    max_tokens: int = 256
    temperature: float = 0.0
    seed: int | None = 1234
    ignore_eos: bool = False  # fixed-output-length variant
    concurrency: int = 1
    load_mode: str = "closed_loop"  # or "poisson"
    request_rate: float | None = None
    timeout_s: float = 600.0
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestResult:
    prompt: Prompt
    send_ts: float
    ttft_s: float | None
    e2e_s: float | None
    chunk_gaps_s: list[float]
    output_text: str
    output_tokens: int
    error: str | None
    spec: dict[str, Any] | None = None  # engine per-request speculative metrics, verbatim
    token_ids: list[int] | None = None  # output token ids, when the engine returns them


def build_body(spec: LoadSpec, p: Prompt) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": spec.model,
        "prompt": p.token_ids,
        "max_tokens": spec.max_tokens,
        "temperature": spec.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if spec.seed is not None:
        body["seed"] = spec.seed
    if spec.ignore_eos:
        body["ignore_eos"] = True  # vLLM/SGLang extension; verify per engine version
    body.update(spec.extra_body)
    return body


def parse_sse_lines(lines: Sequence[str]) -> list[dict[str, Any]]:
    """Parse ``data: {...}`` server-sent-event lines; stops at ``data: [DONE]``."""
    out = []
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        out.append(json.loads(payload))
    return out


async def one_request(client: httpx.AsyncClient, spec: LoadSpec, p: Prompt) -> RequestResult:
    t0 = time.perf_counter()
    send_ts = time.time()
    first: float | None = None
    last = t0
    gaps: list[float] = []
    text: list[str] = []
    usage_tokens: int | None = None
    spec_stats: dict[str, Any] | None = None
    out_ids: list[int] = []
    saw_ids = False
    try:
        async with client.stream(
            "POST", f"{spec.base_url}/v1/completions", json=build_body(spec, p)
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:500]
                return RequestResult(
                    p, send_ts, None, None, [], "", 0, f"HTTP {resp.status_code}: {body}"
                )
            async for line in resp.aiter_lines():
                for ev in parse_sse_lines([line]):
                    now = time.perf_counter()
                    choices = ev.get("choices") or []
                    piece = "".join(c.get("text") or "" for c in choices)
                    for c in choices:  # vLLM return_token_ids: per-chunk delta ids
                        if isinstance(c.get("token_ids"), list):
                            saw_ids = True
                            out_ids.extend(int(t) for t in c["token_ids"])
                    if piece:
                        if first is None:
                            first = now
                        else:
                            gaps.append(now - last)
                        last = now
                        text.append(piece)
                    if ev.get("usage"):
                        usage_tokens = int(ev["usage"].get("completion_tokens", 0))
                    # vLLM >= 0.34 with --per-request-spec-decode-metrics: final chunk carries
                    # metrics.speculative_decoding (num_spec_steps, num_accepted_draft_tokens, ...)
                    m = ev.get("metrics")
                    if isinstance(m, dict) and isinstance(m.get("speculative_decoding"), dict):
                        spec_stats = {"engine": "vllm", **m["speculative_decoding"]}
                    # SGLang >= 0.5.23 with return_spec_tokens_details: response-level sglext chunk
                    sx = ev.get("sglext")
                    if isinstance(sx, dict) and isinstance(sx.get("spec_tokens_details"), dict):
                        spec_stats = {"engine": "sglang", **sx["spec_tokens_details"]}
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return RequestResult(p, send_ts, None, None, gaps, "".join(text), 0, repr(exc))
    end = time.perf_counter()
    if usage_tokens is None:
        return RequestResult(
            p, send_ts, None, end - t0, gaps, "".join(text), 0, "missing usage in stream"
        )
    return RequestResult(
        p,
        send_ts,
        (first - t0) if first is not None else None,
        end - t0,
        gaps,
        "".join(text),
        usage_tokens,
        None,
        spec_stats,
        out_ids if saw_ids else None,
    )


async def run_load(
    spec: LoadSpec,
    prompts: Sequence[Prompt],
    transport: httpx.AsyncBaseTransport | None = None,
    rng_seed: int = 1234,
) -> tuple[list[RequestResult], float]:
    """Run all prompts; returns (results in completion order, wall time in seconds)."""
    limits = httpx.Limits(max_connections=spec.concurrency + 4)
    async with httpx.AsyncClient(
        timeout=spec.timeout_s, limits=limits, transport=transport
    ) as client:
        results: list[RequestResult] = []
        sem = asyncio.Semaphore(spec.concurrency)

        async def guarded(p: Prompt) -> None:
            async with sem:
                results.append(await one_request(client, spec, p))

        t0 = time.perf_counter()
        if spec.load_mode == "closed_loop":
            await asyncio.gather(*(guarded(p) for p in prompts))
        elif spec.load_mode == "poisson":
            if not spec.request_rate or spec.request_rate <= 0:
                raise ValueError("poisson mode needs request_rate > 0")
            rng = random.Random(rng_seed)
            tasks = []
            for p in prompts:
                tasks.append(asyncio.create_task(guarded(p)))
                await asyncio.sleep(rng.expovariate(spec.request_rate))
            await asyncio.gather(*tasks)
        else:
            raise ValueError(f"unknown load_mode {spec.load_mode!r}")
        return results, time.perf_counter() - t0


def to_log_row(
    r: RequestResult,
    spec: LoadSpec,
    *,
    run_id: str,
    arm: str,
    engine: str,
    split: str,
    repeat: int,
    draft_params: dict[str, Any] | None = None,
    spec_counters: Callable[[RequestResult], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tp = (
        tpot_fn(r.e2e_s, r.ttft_s, r.output_tokens)
        if r.e2e_s is not None and r.ttft_s is not None
        else None
    )
    row: dict[str, Any] = {
        "schema_version": "1",
        "run_id": run_id,
        "arm": arm,
        "engine": engine,
        "model": spec.model,
        "language": r.prompt.language,
        "task": r.prompt.task,
        "prompt_id": r.prompt.prompt_id,
        "split": split,
        "concurrency": spec.concurrency,
        "load_mode": spec.load_mode,
        "request_rate": spec.request_rate,
        "repeat": repeat,
        "draft_params": draft_params or {},
        "temperature": spec.temperature,
        "seed": spec.seed,
        "input_tokens": len(r.prompt.token_ids),
        "output_tokens": r.output_tokens,
        "output_chars": len(r.output_text),
        "output_words": len(r.output_text.split()),
        "send_ts": r.send_ts,
        "ttft_s": r.ttft_s,
        "itl_s": r.chunk_gaps_s,
        "tpot_s": None if tp is None or tp != tp else tp,  # NaN -> None
        "e2e_latency_s": r.e2e_s,
        "spec_accepted_tokens": None,
        "spec_proposed_tokens": None,
        "spec_verify_steps": None,
        "acceptance_length": None,
        "spec_metric_source": None,
        "output_token_ids": r.token_ids,
        "error": r.error,
    }
    if r.spec:
        if r.spec.get("engine") == "sglang":
            row.update(spec_fields_from_sglang(r.spec, r.output_tokens))
        else:
            row.update(spec_fields_from_vllm(r.spec))
    if spec_counters is not None:
        row.update(spec_counters(r))
    return row


def spec_fields_from_vllm(sd: dict[str, Any]) -> dict[str, Any]:
    """Map vLLM per-request spec metrics to log fields; tau recomputed as (A + S) / S.

    The engine's own ``mean_acceptance_length`` is kept verbatim so the definitions can be
    cross-checked (spec §8.3: harmonise engine definitions).
    """
    steps = int(sd.get("num_spec_steps") or 0)
    acc = int(sd.get("num_accepted_draft_tokens") or 0)
    return {
        "spec_accepted_tokens": acc,
        "spec_proposed_tokens": int(sd.get("num_draft_tokens") or 0),
        "spec_verify_steps": steps,
        "acceptance_length": (acc + steps) / steps if steps > 0 else None,
        "spec_metric_source": "vllm:per_request_metrics",
        "spec_engine_raw": sd,
    }


def spec_fields_from_sglang(sd: dict[str, Any], output_tokens: int) -> dict[str, Any]:
    """Map SGLang per-request spec details; tau recomputed as (A + S) / S.

    SGLang reports ``spec_verify_ct`` (S) and ``spec_num_correct_drafts`` (A). Its own
    ``spec_accept_length`` is kept verbatim for cross-checking against (A + S) / S and
    against output_tokens / S (the prefill token is not a verify step).
    """
    steps = int(sd.get("spec_verify_ct") or 0)
    acc = int(sd.get("spec_num_correct_drafts") or 0)
    return {
        "spec_accepted_tokens": acc,
        "spec_proposed_tokens": int(sd.get("spec_num_proposed_drafts") or 0),
        "spec_verify_steps": steps,
        "acceptance_length": (acc + steps) / steps if steps > 0 else None,
        "spec_metric_source": "sglang:spec_tokens_details",
        "spec_engine_raw": {**sd, "output_tokens": output_tokens},
    }
