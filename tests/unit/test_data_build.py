"""CPU-only tests for the pure helpers of the WS2 benchmark builder (no downloads)."""

from __future__ import annotations

import json
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from indicspec.data.build import (
    assign_splits,
    classify_passages,
    compare_manifests,
    json_line,
    locate_sentences,
    normalize_ws,
    readme_licence,
    script_fraction,
    seeded_rank,
    text_stats,
)

IDS = st.lists(st.text(min_size=1, max_size=12), min_size=0, max_size=80, unique=True)


@given(IDS, st.randoms())
def test_assign_splits_independent_of_input_order(ids: list[str], rnd) -> None:
    shuffled = list(ids)
    rnd.shuffle(shuffled)
    a = assign_splits(ids, 7, 0.1, 0.2, 30)
    b = assign_splits(shuffled, 7, 0.1, 0.2, 30)
    assert a == b


@given(IDS)
def test_assign_splits_partition_and_sizes(ids: list[str]) -> None:
    res = assign_splits(ids, 1, 0.1, 0.2, 25)
    n = len(ids)
    assert set(res.assignment) | set(res.dropped_by_cap) == set(ids)
    assert not set(res.assignment) & set(res.dropped_by_cap)
    counts = {
        s: sum(v == s for v in res.assignment.values()) for s in ("pilot", "tuning", "report")
    }
    assert counts["pilot"] == int(0.1 * n + 0.5)
    assert counts["tuning"] == int(0.2 * n + 0.5)
    assert counts["report"] == min(25, n - counts["pilot"] - counts["tuning"])


def test_assign_splits_seed_changes_assignment() -> None:
    ids = [f"item{i}" for i in range(100)]
    assert assign_splits(ids, 1, 0.1, 0.2, None) != assign_splits(ids, 2, 0.1, 0.2, None)


def test_assign_splits_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        assign_splits(["a"], 0, 0.6, 0.6, None)


def test_seeded_rank_dedupes_and_is_stable() -> None:
    assert seeded_rank(["b", "a", "b"], 3) == seeded_rank(["a", "b"], 3)
    assert sorted(seeded_rank(["x", "y", "z"], 3, "s")) == ["x", "y", "z"]


def test_locate_sentences_orders_and_drops_overlaps() -> None:
    p = "Alpha beta. Gamma delta. Epsilon."
    assert locate_sentences(p, ["Epsilon.", "Alpha beta.", "Zeta."]) == ["Alpha beta.", "Epsilon."]
    # "beta." overlaps "Alpha beta." which starts earlier -> dropped
    assert locate_sentences(p, ["beta.", "Alpha beta."]) == ["Alpha beta."]
    assert locate_sentences(p, [""]) == []


def test_classify_passages() -> None:
    passages = {"l1": "A. B.", "l2": "C.  D.", "l3": "A. C.", "l4": "Q."}
    out = classify_passages(passages, devtest_en=["A.", "B."], dev_en=["C.", "D."])
    assert out["l1"] == ("devtest", ["A.", "B."])
    assert out["l2"] == ("dev", [])
    assert out["l3"][0] == "mixed"
    assert out["l4"] == ("none", [])


def test_script_fraction_ignores_space_punct_digits() -> None:
    frac, n = script_fraction("தமிழ் 2026, மொழி।", "Tamil")
    assert frac == 1.0 and n > 0
    frac, _ = script_fraction("हिंदी Bush", "Devanagari")
    assert 0 < frac < 1
    assert math.isnan(script_fraction("123 !!", "Telugu")[0])


def test_text_stats_counts_codepoints_words_bytes() -> None:
    assert text_stats("ಕನ್ನಡ ಭಾಷೆ") == {"n_chars": 10, "n_words": 2, "n_bytes": 28}


def test_normalize_ws() -> None:
    assert normalize_ws("  a \t b\n c ") == "a b c"


def test_json_line_is_stable_and_utf8() -> None:
    line = json_line({"b": 1, "a": "मलयाळम"})
    assert line == '{"a":"मलयाळम","b":1}\n'
    assert json.loads(line) == {"a": "मलयाळम", "b": 1}


def test_readme_licence() -> None:
    assert readme_licence("---\nlicense: CC-BY-SA-4.0\n---\n# x") == "cc-by-sa-4.0"
    assert readme_licence("---\r\nlicense: apache-2.0\r\n---\r\n") == "apache-2.0"
    assert readme_licence("# no front matter") is None


def test_compare_manifests_reports_differences() -> None:
    m = {
        "files": {"bench/a.jsonl": {"sha256": "x", "rows": 1}},
        "sources": {"s": {"files_sha256": {"f": "h"}}},
    }
    assert compare_manifests(m, m) == []
    bad = {
        "files": {"bench/a.jsonl": {"sha256": "y", "rows": 1}},
        "sources": {"s": {"files_sha256": {"f": "other"}}},
    }
    problems = compare_manifests(m, bad)
    assert len(problems) == 2
