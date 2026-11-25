"""Load generator tests against an in-process fake streaming server (no GPU, no network)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from indicspec.harness.loadgen import (
    LoadSpec,
    Prompt,
    build_body,
    parse_sse_lines,
    run_load,
    to_log_row,
)
from indicspec.harness.schema import validate


def _sse(chunks: list[str], completion_tokens: int) -> bytes:
    lines = [
        "data: " + json.dumps({"choices": [{"text": c, "index": 0}], "usage": None}) for c in chunks
    ]
    lines.append(
        "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": completion_tokens}})
    )
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def _handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["stream"] is True and isinstance(body["prompt"], list)
    if body["prompt"] == [666]:
        return httpx.Response(500, content=b"boom")
    # two chunks, the second carrying several tokens (as under speculation)
    return httpx.Response(
        200,
        content=_sse(["Hello", " world, again"], 4),
        headers={"content-type": "text/event-stream"},
    )


PROMPTS = [Prompt(f"p{i}", "ta", "qa", [1, 2, 3]) for i in range(5)]


@pytest.mark.parametrize("mode,rate", [("closed_loop", None), ("poisson", 200.0)])
def test_run_load_and_rows_validate(mode, rate):
    spec = LoadSpec(
        "http://fake", "Qwen/Qwen3-4B", concurrency=2, load_mode=mode, request_rate=rate
    )
    results, wall = asyncio.run(run_load(spec, PROMPTS, transport=httpx.MockTransport(_handler)))
    assert len(results) == 5 and wall > 0
    for r in results:
        assert r.error is None
        assert r.output_tokens == 4
        assert r.output_text == "Hello world, again"
        assert r.ttft_s is not None and r.e2e_s is not None and r.e2e_s >= r.ttft_s
        assert len(r.chunk_gaps_s) == 1
        row = to_log_row(r, spec, run_id="t", arm="A0", engine="vllm", split="smoke", repeat=0)
        validate(row, "request_log.v1")
        assert row["output_words"] == 3


def test_http_error_is_recorded_not_raised():
    spec = LoadSpec("http://fake", "m")
    results, _ = asyncio.run(
        run_load(spec, [Prompt("bad", "en", "qa", [666])], transport=httpx.MockTransport(_handler))
    )
    assert results[0].error and results[0].error.startswith("HTTP 500")
    validate(
        to_log_row(results[0], spec, run_id="t", arm="A0", engine="vllm", split="smoke", repeat=0),
        "request_log.v1",
    )


def test_build_body_flags():
    b = build_body(LoadSpec("u", "m", ignore_eos=True, seed=None), PROMPTS[0])
    assert b["ignore_eos"] is True and "seed" not in b
    assert b["stream_options"] == {"include_usage": True}


def test_parse_sse_stops_at_done():
    evs = parse_sse_lines(['data: {"a": 1}', ": comment", "data: [DONE]", 'data: {"b": 2}'])
    assert evs == [{"a": 1}]


def test_poisson_requires_rate():
    with pytest.raises(ValueError):
        asyncio.run(run_load(LoadSpec("u", "m", load_mode="poisson"), PROMPTS[:1]))


def test_vllm_per_request_spec_metrics_are_logged():
    from indicspec.harness.loadgen import spec_fields_from_vllm

    sd = {
        "mean_acceptance_length": 2.5,
        "draft_acceptance_rate": 0.5,
        "acceptance_histogram": [1, 0, 1, 0],
        "num_spec_steps": 2,
        "num_accepted_draft_tokens": 3,
        "num_draft_tokens": 6,
        "num_spec_tokens": 3,
        "per_step_accepted": [0, 3],
        "per_step_drafted": [3, 3],
    }
    f = spec_fields_from_vllm(sd)
    assert f["acceptance_length"] == 2.5 and f["spec_verify_steps"] == 2
    spec = LoadSpec("http://fake", "m")

    def handler(request):
        lines = [
            "data: " + json.dumps({"choices": [{"text": "x"}]}),
            "data: "
            + json.dumps(
                {
                    "choices": [],
                    "usage": {"completion_tokens": 5},
                    "metrics": {"speculative_decoding": sd},
                }
            ),
            "data: [DONE]",
        ]
        return httpx.Response(200, content=("\n\n".join(lines) + "\n\n").encode())

    res, _ = asyncio.run(run_load(spec, PROMPTS[:1], transport=httpx.MockTransport(handler)))
    row = to_log_row(res[0], spec, run_id="t", arm="A5", engine="vllm", split="smoke", repeat=0)
    validate(row, "request_log.v1")
    assert row["acceptance_length"] == 2.5 and row["spec_engine_raw"]["per_step_accepted"] == [0, 3]


def test_sglang_spec_details_are_logged():
    sd = {
        "spec_accept_rate": 0.4,
        "spec_accept_length": 2.0,
        "spec_num_correct_drafts": 6,
        "spec_num_proposed_drafts": 15,
        "spec_verify_ct": 6,
    }
    spec = LoadSpec("http://fake", "m")

    def handler(request):
        lines = [
            "data: " + json.dumps({"choices": [{"text": "abc"}]}),
            "data: " + json.dumps({"choices": [], "sglext": {"spec_tokens_details": sd}}),
            "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": 13}}),
            "data: [DONE]",
        ]
        return httpx.Response(200, content=("\n\n".join(lines) + "\n\n").encode())

    res, _ = asyncio.run(run_load(spec, PROMPTS[:1], transport=httpx.MockTransport(handler)))
    row = to_log_row(res[0], spec, run_id="t", arm="A5", engine="sglang", split="smoke", repeat=0)
    validate(row, "request_log.v1")
    assert row["acceptance_length"] == 2.0 and row["spec_metric_source"].startswith("sglang")
