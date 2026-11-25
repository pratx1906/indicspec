import json
from pathlib import Path

import pytest

from indicspec.analysis.tuning import LANGS, choose, confirm_config, grid


def _req(arm, lang, pid, tps, acc=0, steps=0):
    return {
        "arm": arm,
        "language": lang,
        "prompt_id": pid,
        "concurrency": 1,
        "split": "tuning",
        "error": None,
        "output_tokens": 100,
        "e2e_latency_s": 100 / tps,
        "spec_accepted_tokens": acc,
        "spec_verify_steps": steps,
    }


def test_grid_and_choice(tmp_path: Path):
    blocks = {"A0": ("A0", 10.0), "A1.k2": ("A1.k2", 15.0), "A1.k4": ("A1.k4", 20.0)}
    for name, (arm, tps) in blocks.items():
        d = tmp_path / name
        d.mkdir()
        rows = [_req(arm, lang, f"p{i}", tps, 10, 5) for lang in LANGS for i in range(3)]
        (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    g = grid(tmp_path)
    k4 = [r for r in g if r["setting"] == "A1.k4"]
    assert len(k4) == 6 and all(abs(r["speedup"] - 2.0) < 1e-9 for r in k4)
    assert abs(k4[0]["tau"] - 3.0) < 1e-9
    best = choose(g)
    assert best["A1"]["setting"] == "A1.k4" and abs(best["A1"]["geomean_speedup"] - 2.0) < 1e-9


def test_setting_missing_a_language_cannot_win():
    rows = [
        {"arm": "A5", "setting": "A5.k15", "language": lang, "speedup": 3.0} for lang in LANGS[:5]
    ]
    rows += [{"arm": "A5", "setting": "A5.k7", "language": lang, "speedup": 1.5} for lang in LANGS]
    assert choose(rows)["A5"]["setting"] == "A5.k7"


def test_confirm_config_uses_frozen_design():
    run_cfg = {
        "engine": "vllm",
        "model": "qwen3-4b",
        "tasks": ["qa"],
        "arms": [
            {"arm": "A0"},
            {"arm": "A1", "label": "k4", "spec_override": {"num_speculative_tokens": 4}},
            {"arm": "A5", "label": "k15", "spec_override": {"num_speculative_tokens": 15}},
        ],
    }
    best = {"A1": {"setting": "A1.k4"}, "A5": {"setting": "A5.k15"}}
    cfg = confirm_config(run_cfg, best)
    assert cfg["split"] == "report" and cfg["repeats"] == 3 and cfg["max_prompts_per_cell"] == 100
    assert cfg["concurrency"] == [1, 2, 4, 8, 16, 32]
    assert cfg["arms"][1] == {
        "arm": "A1",
        "label": "k4",
        "spec_override": {"num_speculative_tokens": 4},
    }
    with pytest.raises(ValueError):
        confirm_config(run_cfg, {"A1": {"setting": "A1.k4"}})
