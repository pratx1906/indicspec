from __future__ import annotations

import math
import random

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from indicspec.analysis import metrics as m
from indicspec.analysis.stats import bootstrap_ci, holm, paired_bootstrap_ci
from indicspec.tokens.vocab import coverage


def test_acceptance_length_formula():
    assert m.acceptance_length(0, 10) == 1.0
    assert m.acceptance_length(30, 10) == 4.0
    with pytest.raises(ValueError):
        m.acceptance_length(1, 0)


def test_tpot_and_speed():
    assert m.tpot(1.1, 0.1, 101) == pytest.approx(0.01)
    assert math.isnan(m.tpot(1.0, 0.1, 1))
    assert m.per_user_speed(0.01) == pytest.approx(100.0)


def test_speedup_and_throughput():
    assert m.speedup(200.0, 100.0) == 2.0
    assert m.system_throughput(1000, 4.0) == 250.0


def test_time_per_100_words():
    assert m.time_per_100_words(5.0, 50) == 10.0
    assert math.isnan(m.time_per_100_words(5.0, 0))
    assert m.whitespace_words(" a  b\nc ") == 3


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1, max_size=50))
def test_percentile_matches_numpy_and_is_order_invariant(xs):
    for q in (0, 50, 95, 100):
        assert m.percentile(xs, q) == pytest.approx(float(np.percentile(xs, q)), rel=1e-9, abs=1e-9)
    ys = xs[:]
    random.Random(0).shuffle(ys)
    assert m.percentile(ys, 50) == m.percentile(xs, 50)


def test_cost_model_forms_agree_when_tverify_equals_tar():
    tau, td, tv, to = 3.0, 0.002, 0.01, 0.001
    assert m.predicted_speedup(tau, td, tv, to) == pytest.approx(
        m.predicted_speedup_ar(tau, td, tv, to, t_ar=tv)
    )
    assert m.predicted_speedup(1.0, 0.0, 0.01, 0.0) == 1.0


def test_coverage():
    w, u, n, nu = coverage([1, 1, 1, 2], {1})
    assert (w, u, n, nu) == (0.75, 0.5, 4, 2)
    assert math.isnan(coverage([], {1})[0])


def test_bootstrap_covers_true_mean_of_normal():
    rng = np.random.default_rng(0)
    hits = 0
    trials = 200
    for i in range(trials):
        x = rng.normal(5.0, 2.0, size=60)
        ci = bootstrap_ci(x, n_boot=2000, seed=i)
        hits += ci.lo <= 5.0 <= ci.hi
    # nominal 95%; percentile bootstrap at n=60 is a little anti-conservative
    assert 0.88 <= hits / trials <= 0.99


def test_paired_bootstrap_ratio():
    a = np.full(30, 2.0) + np.linspace(0, 0.1, 30)
    b = np.full(30, 1.0) + np.linspace(0, 0.05, 30)
    ci = paired_bootstrap_ci(a, b, n_boot=2000)
    assert ci.lo <= ci.estimate <= ci.hi
    assert 1.9 < ci.estimate < 2.1
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [1.0, 2.0])


def test_holm():
    assert holm([0.01, 0.04, 0.03], alpha=0.05) == [True, False, False]
    assert holm([0.001, 0.01, 0.02], alpha=0.05) == [True, True, True]
