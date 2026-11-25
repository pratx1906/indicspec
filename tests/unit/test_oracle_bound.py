from hypothesis import given
from hypothesis import strategies as st

from indicspec.analysis.oracle_bound import oracle_steps


def test_all_in_vocab_gives_k_plus_one():
    assert oracle_steps([True] * 8, 3) == [4, 4]


def test_alternating_coverage_caps_runs():
    # lead (out) / cont (in) alternation, as on Dravidian text for a pruned head
    pattern = [False, True] * 4
    st_ = oracle_steps(pattern, 3)
    # step at an out-of-vocab position emits only the bonus token; at an in-vocab one, 1 + bonus
    assert st_ == [1, 2, 2, 2, 1]
    assert sum(st_) == len(pattern)


def test_none_in_vocab_is_autoregressive():
    assert oracle_steps([False] * 5, 7) == [1] * 5


@given(st.lists(st.booleans(), max_size=60), st.integers(1, 8))
def test_steps_cover_sequence_and_respect_cap(mask, k):
    s = oracle_steps(mask, k)
    assert sum(s) == len(mask)
    assert all(1 <= x <= k + 1 for x in s)
