import math

from indicspec.harness.equality_gate import compare_arm, parse_completion, summarise, top2_gap


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
    }
    arm = {"en|a": [1, 2, 3], "ta|b": [1, 7, 3], "ta|c": [1, 7]}
    langs = {"en|a": "en", "ta|b": "ta", "ta|c": "ta", "hi|d": "hi"}
    rows = {r["prompt_id"]: r for r in compare_arm(base, arm, langs)}
    assert rows["en|a"]["status"] == "equal"
    assert rows["ta|b"]["status"] == "near_tie" and rows["ta|b"]["position"] == 1
    assert rows["ta|c"]["status"] == "suspected_bug"
    assert rows["hi|d"]["status"] == "missing"
    s = summarise(list(rows.values()))
    assert s["pass"] is False and s["by_language"]["ta"] == {"near_tie": 1, "suspected_bug": 1}
    assert summarise([rows["en|a"], rows["ta|b"]])["pass"] is True


def test_parse_completion():
    resp = {
        "choices": [{"token_ids": [5, 6], "logprobs": {"top_logprobs": [{"token_id:5": -0.1}]}}]
    }
    assert parse_completion(resp) == ([5, 6], [{"token_id:5": -0.1}])
    assert parse_completion({"choices": [{"token_ids": [1]}]}) == ([1], [])
