import pytest
from hypothesis import given
from hypothesis import strategies as st

from indicspec.analysis.positions import AlignmentError, align, evaluated_positions, split_by


def test_alignment_and_evaluated_positions():
    out = [10, 11, 12, 13, 14, 15, 16]  # 0 = prefill token
    # step0: drafted 3, accepted 2 -> positions 1,2 accepted, 3 rejected; next start 4
    # step1: drafted 3, accepted 3 -> 4,5,6 accepted; next start 8 (> n, truncated by max_tokens)
    ev = evaluated_positions(out, [3, 3], [2, 3])
    assert [(e.position, e.accepted) for e in ev] == [
        (1, True),
        (2, True),
        (3, False),
        (4, True),
        (5, True),
        (6, True),
    ]
    assert ev[2].token_id == 13 and ev[2].depth == 2


def test_alignment_rejects_inconsistent_counts():
    with pytest.raises(AlignmentError):
        align(10, [3], [1])  # covers only 3 of 10 positions
    with pytest.raises(AlignmentError):
        align(5, [2], [3])  # accepted > drafted


def test_split_by_vocab_and_class():
    out = [0, 1, 2, 3, 4]
    ev = evaluated_positions(out, [2, 2], [1, 1])
    # step0 p=1: pos1 acc, pos2 rej; step1 p=3: pos3 acc, pos4 rej
    t = split_by(ev, key={1: "CONT", 2: "LEAD", 3: "CONT", 4: "LEAD"}, vocab={1, 3})
    assert t["CONT|in_vocab"].rate == 1.0 and t["CONT|in_vocab"].evaluated == 2
    assert t["LEAD|out_vocab"].rate == 0.0 and t["LEAD|out_vocab"].evaluated == 2


@given(st.lists(st.tuples(st.integers(1, 6), st.integers(0, 6)), min_size=1, max_size=20))
def test_generated_steps_always_align(steps):
    steps = [(d, min(a, d)) for d, a in steps]
    n = 1 + sum(a + 1 for _, a in steps)
    ev = evaluated_positions(list(range(n)), [d for d, _ in steps], [a for _, a in steps])
    assert sum(e.accepted for e in ev) == sum(a for _, a in steps)
    assert all(0 < e.position < n for e in ev)
