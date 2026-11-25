from __future__ import annotations

import math

from indicspec.analysis.tokentax import parity, unique_refs


def test_parity_ratio_and_median():
    en = {"a": 10, "b": 20}
    xx = {"a": 60, "b": 100, "c": 5}  # "c" has no English partner -> ignored
    ros, med, lo, hi, n = parity(xx, en)
    assert n == 2
    assert ros == 160 / 30
    assert med == 5.5  # per-item ratios 6.0 and 5.0
    assert lo <= ros <= hi


def test_parity_no_overlap_is_nan():
    assert math.isnan(parity({"x": 1}, {"y": 1})[0])


def test_unique_refs_dedups_by_task_lang_item():
    rows = [
        {"task": "t", "lang": "en", "item_id": 1, "reference_text": "a"},
        {"task": "t", "lang": "en", "item_id": 1, "reference_text": "b"},
        {"task": "t", "lang": "hi", "item_id": 1, "reference_text": "c"},
    ]
    assert unique_refs(rows) == {("t", "en", "1"): "a", ("t", "hi", "1"): "c"}
