"""Confirmatory analysis on synthetic runs (no GPU, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from indicspec.analysis import confirm as cf

LANGS = ("en", "hi", "ta", "te", "kn", "ml")


def _req(arm, model, lang, conc, rep, item, tau, tps=50.0, split="report", **extra):
    steps = 100
    acc = round((tau - 1) * steps)
    return {
        "arm": arm,
        "model": model,
        "engine": "vllm",
        "language": lang,
        "concurrency": conc,
        "repeat": rep,
        "prompt_id": f"qa:{item}",
        "split": split,
        "error": None,
        "spec_accepted_tokens": acc if arm != "A0" else None,
        "spec_verify_steps": steps if arm != "A0" else None,
        "output_tokens": 256,
        "e2e_latency_s": 256 / tps,
        **extra,
    }


def _write_block(run: Path, name: str, rows, cells=None):
    b = run / name
    b.mkdir(parents=True, exist_ok=True)
    (b / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if cells is not None:
        (b / "cells.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cells))


def _cells(lang_conc_rep, power=175.0, n_err=0):
    return [
        {
            "cell": f"{la}|c{c}|r{r}",
            "language": la,
            "concurrency": c,
            "repeat": r,
            "start_unix": 0.0,
            "end_unix": 1.0,
            "n_prompts": 20,
            "n_errors": n_err,
            "enforced_power_limit_w": power,
        }
        for la, c, r in lang_conc_rep
    ]


def _synthetic_run(
    tmp: Path,
    arm: str,
    tau_by_lang: dict[str, float],
    model="qwen3-8b",
    concs=(1,),
    tps_by=None,
    n=30,
    jitter=0.05,
):
    rng = np.random.default_rng(0)
    rows, a0 = [], []
    keys = []
    for la in LANGS:
        for c in concs:
            for r in range(2):
                keys.append((la, c, r))
                for i in range(n):
                    t = max(1.0, tau_by_lang[la] + rng.normal(0, jitter))
                    sp = (tps_by or {}).get((la, c), 2.0)
                    rows.append(_req(arm, model, la, c, r, i, t, tps=50.0 * sp))
                    a0.append(_req("A0", model, la, c, r, i, 1.0, tps=50.0))
    _write_block(tmp, f"vllm-{arm}", rows, _cells(keys))
    _write_block(tmp, "vllm-A0", a0, _cells(keys))
    return tmp


def test_rejects_non_report_split(tmp_path: Path):
    _write_block(tmp_path, "b", [_req("A5", "m", "en", 1, 0, 1, 2.0, split="pilot")])
    with pytest.raises(ValueError, match="report split"):
        cf.load([tmp_path])


def test_tau_and_ratio():
    rows = [_req("A5", "m", "en", 1, r, i, 3.0) for r in range(2) for i in range(5)]
    rows += [_req("A5", "m", "ta", 1, r, i, 1.5) for r in range(2) for i in range(5)]
    tau = cf.per_prompt(rows, cf.request_tau)
    assert tau[("A5", "m", "vllm", "en", 1)]["qa:0"] == pytest.approx(3.0)
    r, boots, n = cf.boot_ratio(
        tau[("A5", "m", "vllm", "ta", 1)], tau[("A5", "m", "vllm", "en", 1)]
    )
    assert n == 5 and r == pytest.approx(0.5) and boots.size == cf.N_BOOT


def test_h1_confirmed_on_clear_collapse(tmp_path: Path):
    taus = {"en": 2.5, "hi": 1.4, "ta": 1.1, "te": 1.15, "kn": 1.1, "ml": 1.1}
    d = cf.load([_synthetic_run(tmp_path, "A3", taus)])
    _, verdicts = cf.run_all(d)
    pair = verdicts["H1"]["pairs"]["A3@qwen3-8b"]
    assert pair["verdict"] == "confirmed"
    assert all(v == "confirmed" for v in pair["per_language"].values())
    assert verdicts["H1"]["verdict"] == "confirmed"


def test_h1_falsified_without_gap(tmp_path: Path):
    taus = dict.fromkeys(LANGS, 2.4)
    d = cf.load([_synthetic_run(tmp_path, "A3", taus)])
    _, verdicts = cf.run_all(d)
    assert verdicts["H1"]["pairs"]["A3@qwen3-8b"]["verdict"] == "falsified"


def test_h2_confirmed_when_no_gap_and_g1_no(tmp_path: Path):
    taus = dict.fromkeys(LANGS, 3.0)
    d = cf.load([_synthetic_run(tmp_path, "A5", taus, model="qwen3-4b")])
    _, verdicts = cf.run_all(d)
    assert verdicts["H2"]["pairs"]["A5@qwen3-4b"]["verdict"] == "confirmed"
    assert verdicts["G1"]["gate"] == "NO"


def test_h2_falsified_and_g1_yes_on_gap(tmp_path: Path):
    taus = {"en": 3.0, "hi": 2.5, "ta": 1.5, "te": 1.4, "kn": 1.5, "ml": 1.3}
    d = cf.load([_synthetic_run(tmp_path, "A5", taus, model="qwen3-4b")])
    _, verdicts = cf.run_all(d)
    assert verdicts["H2"]["pairs"]["A5@qwen3-4b"]["verdict"] == "falsified"
    assert verdicts["G1"]["gate"] == "YES"


def test_holm():
    assert cf.holm_reject({"a": 0.001, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert cf.holm_reject({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}


def test_h4_crossover(tmp_path: Path):
    concs = (1, 2, 4, 8, 16)
    sp = {}
    for la in LANGS:
        for c in concs:
            if la in ("en", "hi"):
                sp[(la, c)] = 2.0  # never crosses
            else:
                sp[(la, c)] = 2.0 if c < 4 else 0.7  # crosses at 4 (grid index 2)
    taus = dict.fromkeys(LANGS, 2.0)
    d = cf.load([_synthetic_run(tmp_path, "A1", taus, model="qwen3-4b", concs=concs, tps_by=sp)])
    _, verdicts = cf.run_all(d)
    pair = verdicts["H4"]["pairs"]["A1@qwen3-4b"]
    assert pair["crossover"]["en"] == ">16" and pair["crossover"]["ta"] == 4
    assert pair["verdict"] == "confirmed"  # index 2 vs 5: >= 2 grid steps


def test_low_power_and_failed_cells_are_excluded_and_listed(tmp_path: Path):
    rows = [_req("A5", "m", "en", 1, 0, i, 2.0) for i in range(5)]
    rows += [_req("A5", "m", "en", 1, 1, i, 2.0) for i in range(5)]
    cells = _cells([("en", 1, 0)], power=55.0) + _cells([("en", 1, 1)], n_err=5)
    _write_block(tmp_path, "b", rows, cells)
    kept, excluded = cf.apply_exclusions(cf.load([tmp_path]))
    assert kept == []
    reasons = {(e["repeat"]): e["reasons"] for e in excluded}
    assert any("55 W" in r for r in reasons[0]) and any("failed" in r for r in reasons[1])


def test_hot_cell_excluded(tmp_path: Path):
    cell = _cells([("en", 1, 0)])[0]
    tele = [{"t": 0.5, "sm": 2000.0, "temp": 97.0, "mem": 20000.0, "util": 99.0}]
    reasons = cf.cell_exclusion_reasons(cell, tele, 24463)
    assert any("temperature" in r for r in reasons)
    tele = [{"t": 0.5, "sm": 500.0, "temp": 80.0, "mem": 20000.0, "util": 99.0}] + [
        {"t": 10.0 + i, "sm": 2000.0, "temp": 80.0, "mem": 20000.0, "util": 99.0} for i in range(20)
    ]
    assert any("SM clock" in r for r in cf.cell_exclusion_reasons(cell, tele, 24463))


def test_telemetry_parse(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text(
        "timestamp, clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W], "
        "temperature.gpu, memory.used [MiB], utilization.gpu [%], clocks_event_reasons.active\n"
        "2026/11/25 18:27:56.257, 1597, 14001, 33.15, 65, 1636, 3, 0x0\n"
    )
    rows = cf.parse_telemetry(p, utc_offset_h=0)
    assert rows[0]["sm"] == 1597 and rows[0]["temp"] == 65 and rows[0]["util"] == 3
    assert rows[0]["t"] == pytest.approx(1795631276.257, abs=1e-3)


def test_h3_and_g2_on_positions(tmp_path: Path):
    # en tokens (class ASCII) accepted 80%; Dravidian CONT accepted 90%, LEAD 10%
    cls = {1: "ASCII_OR_PUNCT", 2: "BYTE_FRAGMENT_LEAD", 3: "BYTE_FRAGMENT_CONT"}
    rows = []
    rng = np.random.default_rng(1)
    for la in LANGS:
        for i in range(20):
            ids, drafted, accepted = [0], [], []
            for _ in range(10):
                if la == "en":
                    a = int(rng.random() < 0.8)
                    toks = [1, 1]
                else:
                    a = int(rng.random() < 0.1)  # lead rarely accepted ...
                    toks = [2, 3]
                    if a:  # ... and cont usually follows
                        a += int(rng.random() < 0.9)
                drafted.append(2)
                accepted.append(min(a, 2))
                ids += [*toks[: min(a, 2)], toks[min(a, 1)]]
            rows.append(
                _req(
                    "A5",
                    "qwen3-4b",
                    la,
                    1,
                    0,
                    i,
                    2.0,
                    output_token_ids=ids,
                    spec_engine_raw={"per_step_drafted": drafted, "per_step_accepted": accepted},
                )
            )
    h3 = cf.h3(rows, cls)
    assert h3["pairs"]["A5@qwen3-4b"]["aligned_fraction"] == 1.0
    assert h3["pairs"]["A5@qwen3-4b"]["verdict"] in ("confirmed", "falsified", "inconclusive")
    g2 = cf.g2(rows, cls)
    assert g2["pairs"]["A5@qwen3-4b"]["a"] is True  # LEAD vs CONT clearly differ


def test_g3_rule():
    assert cf.g3({"ta": 1.1, "te": 1.2, "kn": 1.06, "ml": 1.0}) == "YES"
    assert cf.g3({"ta": 1.1, "te": 1.0, "kn": 1.04, "ml": 1.0}) == "NO"
