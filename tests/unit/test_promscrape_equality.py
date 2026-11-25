from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from indicspec.harness.equality import classify, compare, first_divergence
from indicspec.harness.promscrape import delta, parse, totals

TEXT = """# HELP spec_accepted Accepted draft tokens
# TYPE spec_accepted counter
spec_accepted_total{model_name="Qwen/Qwen3-4B",engine="0"} 120.0
spec_accepted_total{model_name="Qwen/Qwen3-4B",engine="1"} 30
spec_drafted_total{model_name="Qwen/Qwen3-4B"} 400 1700000000000
weird_gauge NaN
hist_bucket{le="+Inf"} 7
"""


def test_parse_and_totals():
    s = parse(TEXT)
    assert len(s) == 5
    t = totals(s, ["spec_accepted_total", "spec_drafted_total", "absent"])
    assert t == {"spec_accepted_total": 150.0, "spec_drafted_total": 400.0, "absent": 0.0}
    assert math.isnan(next(x.value for x in s if x.name == "weird_gauge"))
    assert ("le", "+Inf") in next(x.labels for x in s if x.name == "hist_bucket")


def test_delta_and_reset_detection():
    assert delta({"a": 1.0}, {"a": 4.0}) == {"a": 3.0}
    with pytest.raises(ValueError):
        delta({"a": 5.0}, {"a": 1.0})


def test_first_divergence_cases():
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    d = compare("p", [1, 2], [1, 2, 3])
    assert d is not None and d.baseline_token is None and d.arm_token == 3
    assert compare("p", [4], [4]) is None


@given(st.lists(st.integers(0, 50), max_size=30), st.lists(st.integers(0, 50), max_size=30))
def test_divergence_prefix_property(a, b):
    pos = first_divergence(a, b)
    if pos is None:
        assert a == b
    else:
        assert a[:pos] == b[:pos]
        assert pos == min(len(a), len(b)) or a[pos] != b[pos]


def test_classify():
    assert classify(0.01, 0.1) == "near_tie"
    assert classify(2.0, 0.1) == "suspected_bug"
