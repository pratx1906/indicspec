from __future__ import annotations

import json

import pytest

from indicspec.harness.server import build_plan, load_configs


def _plan(engine: str, arm: str, **kw):
    return build_plan(
        engine,
        arm,
        model="qwen3-4b",
        target_path="/m/target",
        drafter_path="/m/draft",
        **kw,
    )


def test_configs_load_and_cover_arms():
    engines, arms, models = load_configs()
    assert {"vllm", "sglang"} <= engines.keys()
    assert {"A0", "A1", "A3", "A4", "A5"} <= arms["arms"].keys()
    assert "qwen3-8b" in models["targets"]


def test_vllm_a0_has_no_spec_config():
    p = _plan("vllm", "A0")
    assert "--speculative-config" not in p.cmd
    assert p.env["VLLM_WSL2_ENABLE_PIN_MEMORY"] == "1"
    assert p.cmd[1:3] == ["serve", "/m/target"]


def test_vllm_dflash_spec_json_and_override():
    p = _plan("vllm", "A5", overrides={"num_speculative_tokens": 7})
    spec = json.loads(p.cmd[p.cmd.index("--speculative-config") + 1])
    assert spec == {"method": "dflash", "model": "/m/draft", "num_speculative_tokens": 7}
    assert "--per-request-spec-decode-metrics" in p.cmd


def test_sglang_eagle3_flags_and_no_fa3():
    p = _plan("sglang", "A3")
    assert p.cmd[p.cmd.index("--speculative-algorithm") + 1] == "EAGLE3"
    assert p.cmd[p.cmd.index("--speculative-draft-model-path") + 1] == "/m/draft"
    assert p.cmd[p.cmd.index("--attention-backend") + 1] != "fa3"
    with pytest.raises(ValueError):
        _plan("sglang", "A3", overrides={"x": 1})
