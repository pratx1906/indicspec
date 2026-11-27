from indicspec.analysis.digest import speedup_table, verdict_lines


def test_verdict_lines_and_table():
    d = {
        "prereg_commit": "abc",
        "n_requests_kept": 10,
        "excluded_cells": [],
        "H2": {
            "pairs": {"A1@m": {"verdict": "confirmed", "per_language": {"ta": "confirmed"}}},
            "verdict": "confirmed",
        },
        "G1": {"pairs": {"A1@m": {"hit": True}}, "gate": "YES"},
        "G3": "not evaluated",
    }
    text = "\n".join(verdict_lines(d))
    assert "**H2**: confirmed" in text and "per_language ta confirmed" in text
    assert "**G1**: YES" in text and "**G3**: not evaluated" in text
    rows = [
        {"model": "m", "arm": "A0", "language": "en", "concurrency": "1", "speedup": ""},
        {
            "model": "m",
            "arm": "A1",
            "language": "en",
            "concurrency": "1",
            "speedup": "1.5",
            "speedup_ci_lo": "1.4",
            "speedup_ci_hi": "1.6",
        },
    ]
    t = "\n".join(speedup_table(rows, "m"))
    assert "| A1 | 1.50 (1.40–1.60) | – |" in t
