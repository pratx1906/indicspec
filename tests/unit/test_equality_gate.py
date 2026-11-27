import math

from indicspec.harness.equality_gate import (
    amended_status,
    compare_arm,
    parse_completion,
    summarise,
    token_rank,
    top2_gap,
)


def test_top2_gap():
    gap = top2_gap({"token_id:1": -0.1, "token_id:2": -0.15, "token_id:3": -3.0})
    assert abs(gap - 0.05) < 1e-12
    assert math.isinf(top2_gap({"token_id:1": -0.1}))
    assert math.isinf(top2_gap(None))


def test_compare_arm_statuses():
    base = {
        "en|a": ([1, 2, 3], [{"t:1": 0.0, "t:9": -5.0}] * 3),
        "ta|b": ([1, 2, 3], [None, {"t:2": -0.30, "t:7": -0.35}, None]),
        "ta|c": ([1, 2, 3], [None, {"t:2": -0.1, "t:7": -4.0}, None]),
        "hi|d": ([1], [None]),
        "te|e": ([4, 5], [{"t:4": -0.2, "t:8": -0.7, "t:6": -2.0}, None]),
        "kn|f": ([4, 5], [{"t:4": -0.2, "t:8": -0.7, "t:6": -2.0}, None]),
    }
    arm = {"en|a": [1, 2, 3], "ta|b": [1, 7, 3], "ta|c": [1, 7], "te|e": [8, 5], "kn|f": [9, 5]}
    langs = {"en|a": "en", "ta|b": "ta", "ta|c": "ta", "hi|d": "hi", "te|e": "te", "kn|f": "kn"}
    rows = {r["prompt_id"]: r for r in compare_arm(base, arm, langs)}
    assert rows["en|a"]["status"] == "equal"
    assert rows["ta|b"]["status"] == "near_tie" and rows["ta|b"]["position"] == 1
    assert rows["ta|c"]["status"] == "suspected_bug"
    assert rows["hi|d"]["status"] == "missing"
    # gap 0.5, arm token is the target's rank 2: near-tie (Amendment 2), bug under frozen 0.1
    assert rows["te|e"]["status"] == "near_tie" and rows["te|e"]["frozen_status"] == "suspected_bug"
    assert rows["te|e"]["arm_token_rank"] == 2
    # same gap but the arm token is not among the target's candidates
    assert rows["kn|f"]["status"] == "suspected_bug" and rows["kn|f"]["arm_token_rank"] is None
    s = summarise(list(rows.values()))
    assert s["pass"] is False and s["by_language"]["ta"] == {"near_tie": 1, "suspected_bug": 1}
    assert summarise([rows["en|a"], rows["ta|b"]])["pass"] is True


def test_parse_completion():
    resp = {
        "choices": [{"token_ids": [5, 6], "logprobs": {"top_logprobs": [{"token_id:5": -0.1}]}}]
    }
    assert parse_completion(resp) == ([5, 6], [{"token_id:5": -0.1}])
    assert parse_completion({"choices": [{"token_ids": [1]}]}) == ([1], [])


def test_amended_rule_edges():
    assert amended_status(1.0, 3) == "near_tie"
    assert amended_status(1.125, 2) == "suspected_bug"
    assert amended_status(0.125, 4) == "suspected_bug"
    assert amended_status(0.0, None) == "suspected_bug"
    assert token_rank({"token_id:3": -1.0, "token_id:9": -0.5}, 3) == 2
    assert token_rank(None, 3) is None
