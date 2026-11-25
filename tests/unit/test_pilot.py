import json
from pathlib import Path

from indicspec.analysis.pilot import analyse, needed_n


def _req(arm, lang, conc, rep, pid, acc, steps, out_tok, e2e):
    return {
        "arm": arm,
        "language": lang,
        "concurrency": conc,
        "repeat": rep,
        "prompt_id": pid,
        "spec_accepted_tokens": acc,
        "spec_verify_steps": steps,
        "acceptance_length": (acc + steps) / steps if steps else None,
        "output_tokens": out_tok,
        "e2e_latency_s": e2e,
        "error": None,
        "output_words": 10,
    }


def test_analyse_tau_and_paired_speedup(tmp_path: Path):
    run = tmp_path / "run"
    for arm, acc, e2e in (("A0", 0, 2.0), ("A5", 30, 1.0)):
        b = run / f"vllm-{arm}"
        b.mkdir(parents=True)
        rows = []
        for rep in range(2):
            for i in range(4):
                steps = 10 if arm == "A5" else 0
                rows.append(_req(arm, "ta", 1, rep, f"p{i}", acc + i, steps, 40, e2e + 0.01 * i))
        (b / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        cells = [
            {
                "cell": f"ta|c1|r{rep}",
                "language": "ta",
                "concurrency": 1,
                "repeat": rep,
                "wall_s": 8.0,
                "energy_mj": 400_000,
                "total_output_tokens": 160,
            }
            for rep in range(2)
        ]
        (b / "cells.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cells))
    rows, thermal = analyse(run, tau_hw=0.1)
    a5 = next(r for r in rows if r["arm"] == "A5")
    assert abs(a5["tau"] - (sum(30 + i for i in range(4)) * 2 + 80) / 80) < 1e-9
    assert 1.9 < a5["speedup"] < 2.1  # ~half the latency
    assert abs(a5["joules_per_token"] - 800 / 320) < 1e-9
    assert thermal == []


def test_needed_n():
    assert needed_n(0.5, 0.1) == 97
    assert needed_n(0.0, 0.1) is None
