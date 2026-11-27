from indicspec.analysis.breakdown import by_task, energy


def _req(arm, lang, task, pid, tps, rep=0, words=10, acc=0, steps=0):
    return {
        "arm": arm,
        "language": lang,
        "task": task,
        "prompt_id": pid,
        "concurrency": 1,
        "repeat": rep,
        "error": None,
        "output_tokens": 100,
        "output_words": words,
        "e2e_latency_s": 100 / tps,
        "spec_accepted_tokens": acc,
        "spec_verify_steps": steps,
    }


def test_by_task_pairs_by_prompt_and_averages_repeats():
    rows = [
        _req("A0", "ta", "qa", "p1", 10.0),
        _req("A0", "ta", "qa", "p1", 10.0, rep=1),
        _req("A1", "ta", "qa", "p1", 20.0, acc=6, steps=3),
        _req("A1", "ta", "qa", "p1", 40.0, rep=1, acc=6, steps=3),
        _req("A1", "ta", "qa", "p2", 99.0),  # no A0 partner: ignored for speed-up
        _req("A0", "ta", "chat", "p3", 10.0),
        _req("A1", "ta", "chat", "p3", 15.0),
    ]
    out = {(r["task"]): r for r in by_task(rows)}
    assert out["qa"]["n_items"] == 1 and abs(out["qa"]["speedup"] - 3.0) < 1e-9
    assert abs(out["qa"]["tau"] - 3.0) < 1e-9
    assert abs(out["chat"]["speedup"] - 1.5) < 1e-9 and out["chat"]["tau"] is None


def test_energy_per_token_and_word_relative_to_a0():
    rows = [
        _req("A0", "ml", "qa", "p", 10.0, words=50),
        _req("A1", "ml", "qa", "p", 20.0, words=50),
    ]
    cells = [
        (
            "A0",
            {
                "language": "ml",
                "concurrency": 1,
                "repeat": 0,
                "energy_mj": 200_000,
                "total_output_tokens": 100,
            },
        ),
        (
            "A1",
            {
                "language": "ml",
                "concurrency": 1,
                "repeat": 0,
                "energy_mj": 100_000,
                "total_output_tokens": 100,
            },
        ),
    ]
    out = {r["arm"]: r for r in energy(cells, rows)}
    assert abs(out["A0"]["joules_per_1k_tokens"] - 2000.0) < 1e-9
    assert abs(out["A0"]["joules_per_100_words"] - 400.0) < 1e-9
    assert abs(out["A1"]["energy_vs_A0"] - 0.5) < 1e-9
