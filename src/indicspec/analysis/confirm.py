"""Confirmatory analysis exactly as pre-registered in docs/PREREG.md (frozen at 6023a78).

Reads ONLY ``report``-split request rows from ``results/raw/<run>/<block>/requests.jsonl``;
any other split in the input is an error. Applies the PREREG §5 cell-exclusion rules (power,
thermal, VRAM, failed requests) at cell level and lists every excluded cell in the verdicts file.

Outputs ``results/tables/<name>_confirm.csv`` (per arm x model x engine x language x
concurrency) and ``results/tables/<name>_verdicts.json`` (H1-H4, G1-G3, exclusions, and every
reading of an ambiguous PREREG clause, see ``INTERPRETATIONS``).

Usage: python -m indicspec.analysis.confirm --run confirm_vllm_4b --run confirm_vllm_8b
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from indicspec.analysis.positions import AlignmentError, evaluated_positions

REPO = Path(__file__).resolve().parents[3]
DRAVIDIAN = ("ta", "te", "kn", "ml")
LANGS = ("en", "hi", *DRAVIDIAN)
N_BOOT = 10_000
SEED = 1234
ALPHA = 0.05
VRAM_TOTAL_MIB_DEFAULT = 24463
VRAM_MARGIN_MIB = 512
MIN_POWER_W = 140.0
MAX_TEMP_C = 95.0
MIN_CLOCK_RATIO = 0.6
MAX_FAIL_FRACTION = 0.05

# Where PREREG text admits more than one reading, the strictest is implemented and recorded here.
INTERPRETATIONS = {
    "holm_plus_ci": "A per-language confirm/falsify requires BOTH the plain 95% CI condition and "
    "rejection under Holm (family-wise alpha 0.05 across ta/te/kn/ml) of the one-sided bootstrap "
    "p-value (fraction of resamples on the wrong side of the threshold).",
    "h1_falsify_tau_en": "H1 falsification compares tau_L's lower CI with 0.9 x the point estimate "
    "of tau_en for the same head (as written).",
    "multi_arm_overall": "For hypotheses covering several (arm, model) pairs, the overall verdict "
    "is "
    "'confirmed' only if every pair is confirmed, 'falsified' only if every pair is falsified, "
    "otherwise 'inconclusive'.",
    "h3_pooled": "H3 pools ta/te/kn/ml into one test per (arm, model), so the 3-of-4 language rule "
    "does not apply; verdict per pair, overall by the multi-arm rule. CI by cluster bootstrap over "
    "requests.",
    "h4_bonferroni": "H4 crossover uses one-sided speed-up upper bounds at level 0.05/4 "
    "(Bonferroni "
    "across the four Dravidian languages), stricter than Holm.",
    "g2b_upper_ci": "G2(b) requires the upper 95% CI of (class-standardised gap / raw gap) to be "
    "<= 0.5, and a positive raw gap; classes absent from Dravidian positions are dropped and the "
    "English weights renormalised.",
    "thermal_busy_samples": "Thermal clock rule uses busy samples (utilisation > 50%): the cell's "
    "5th-percentile SM clock vs the block's busy-sample median.",
    "power_missing": "A cell without a recorded enforced power limit is excluded (power unknown).",
    "telemetry_time": "nvidia-smi timestamps are local wall time; they are converted with the "
    "analysing machine's local time zone (or --telemetry-utc-offset).",
}


# --------------------------------------------------------------------------- loading


@dataclass
class Data:
    requests: list[dict[str, Any]] = field(default_factory=list)
    cells: list[dict[str, Any]] = field(default_factory=list)
    telemetry: dict[str, list[dict[str, float]]] = field(default_factory=dict)  # block -> rows


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def parse_telemetry(path: Path, utc_offset_h: float | None = None) -> list[dict[str, float]]:
    """nvidia-smi CSV -> rows with unix time, sm clock, temp, memory, utilisation."""
    out: list[dict[str, float]] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return out
        col = {h.strip().split(" [")[0]: i for i, h in enumerate(header)}
        clock_col = col.get("clocks.sm", col.get("clocks.current.sm"))
        for row in reader:
            if len(row) < len(col):
                continue
            try:
                ts = dt.datetime.strptime(row[col["timestamp"]].strip(), "%Y/%m/%d %H:%M:%S.%f")
                if utc_offset_h is None:
                    unix = ts.timestamp()
                else:
                    unix = ts.replace(
                        tzinfo=dt.timezone(dt.timedelta(hours=utc_offset_h))
                    ).timestamp()
                out.append(
                    {
                        "t": unix,
                        "sm": float(row[clock_col].strip()) if clock_col is not None else np.nan,
                        "temp": float(row[col["temperature.gpu"]].strip()),
                        "mem": float(row[col["memory.used"]].strip()),
                        "util": float(row[col["utilization.gpu"]].strip())
                        if "utilization.gpu" in col
                        else 100.0,
                    }
                )
            except (ValueError, KeyError):
                continue
    return out


def load(run_dirs: Iterable[Path], utc_offset_h: float | None = None) -> Data:
    d = Data()
    for run in run_dirs:
        for block in sorted(p for p in run.iterdir() if p.is_dir()):
            bname = f"{run.name}/{block.name}"
            for r in _rows(block / "requests.jsonl"):
                if r.get("split") != "report":
                    raise ValueError(
                        f"confirmatory analysis reads only the report split; {bname} has "
                        f"split={r.get('split')!r}"
                    )
                r["_block"] = bname
                d.requests.append(r)
            for c in _rows(block / "cells.jsonl"):
                c["_block"] = bname
                d.cells.append(c)
            tele: list[dict[str, float]] = []
            for t in sorted(block.glob("telemetry-*.csv")):
                tele += parse_telemetry(t, utc_offset_h)
            d.telemetry[bname] = sorted(tele, key=lambda x: x["t"])
    return d


# --------------------------------------------------------------------------- exclusions


def cell_exclusion_reasons(
    cell: Mapping[str, Any], telemetry: Sequence[Mapping[str, float]], vram_total_mib: int
) -> list[str]:
    reasons = []
    p = cell.get("enforced_power_limit_w")
    if p is None:
        reasons.append("power unknown")
    elif float(p) < MIN_POWER_W:
        reasons.append(f"enforced power {float(p):.0f} W < {MIN_POWER_W:.0f} W")
    n = int(cell.get("n_prompts") or 0)
    ne = int(cell.get("n_errors") or 0)
    if n and ne / n > MAX_FAIL_FRACTION:
        reasons.append(f"failed requests {ne}/{n} > 5%")
    if telemetry:
        t0, t1 = float(cell["start_unix"]), float(cell["end_unix"])
        win = [s for s in telemetry if t0 <= s["t"] <= t1]
        busy_block = [s["sm"] for s in telemetry if s["util"] > 50 and not np.isnan(s["sm"])]
        if win:
            if max(s["temp"] for s in win) > MAX_TEMP_C:
                reasons.append(f"temperature > {MAX_TEMP_C:.0f} C")
            if max(s["mem"] for s in win) >= vram_total_mib - VRAM_MARGIN_MIB:
                reasons.append("VRAM peak at ceiling")
            busy_win = [s["sm"] for s in win if s["util"] > 50 and not np.isnan(s["sm"])]
            if busy_win and busy_block:
                p5 = float(np.percentile(busy_win, 5))
                med = float(np.median(busy_block))
                if p5 < MIN_CLOCK_RATIO * med:
                    reasons.append(
                        f"p5 SM clock {p5:.0f} < {MIN_CLOCK_RATIO} x block median {med:.0f}"
                    )
    return reasons


def apply_exclusions(
    d: Data, vram_total_mib: int = VRAM_TOTAL_MIB_DEFAULT
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (kept request rows without errors, excluded-cell records)."""
    excluded: dict[tuple[str, str, int, int], list[str]] = {}
    for c in d.cells:
        reasons = cell_exclusion_reasons(c, d.telemetry.get(c["_block"], []), vram_total_mib)
        if reasons:
            excluded[(c["_block"], c["language"], int(c["concurrency"]), int(c["repeat"]))] = (
                reasons
            )
    kept = []
    for r in d.requests:
        key = (r["_block"], r["language"], int(r["concurrency"]), int(r["repeat"]))
        if key in excluded or r.get("error"):
            continue
        kept.append(r)
    records = [
        {"block": k[0], "language": k[1], "concurrency": k[2], "repeat": k[3], "reasons": v}
        for k, v in sorted(excluded.items())
    ]
    return kept, records


# --------------------------------------------------------------------------- per-prompt values


def arm_base(arm: str) -> str:
    return arm.split(".")[0]


def request_tau(r: Mapping[str, Any]) -> float | None:
    s = r.get("spec_verify_steps") or 0
    if not s:
        return None
    return (float(r.get("spec_accepted_tokens") or 0) + s) / s


def request_tps(r: Mapping[str, Any]) -> float | None:
    e2e = r.get("e2e_latency_s")
    if not e2e or not r.get("output_tokens"):
        return None
    return float(r["output_tokens"]) / float(e2e)


Key = tuple[str, str, str, str, int]  # arm, model, engine, language, concurrency


def per_prompt(
    rows: Iterable[Mapping[str, Any]], value: Callable[[Mapping[str, Any]], float | None]
) -> dict[Key, dict[str, float]]:
    """Mean over repeats of a per-request value, per (arm, model, engine, language, conc, item)."""
    acc: dict[Key, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = value(r)
        if v is None:
            continue
        k: Key = (r["arm"], r["model"], r["engine"], r["language"], int(r["concurrency"]))
        acc[k][r["prompt_id"]].append(v)
    return {k: {p: float(np.mean(v)) for p, v in items.items()} for k, items in acc.items()}


# --------------------------------------------------------------------------- bootstrap


def _boot_idx(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(N_BOOT, n))


def boot_mean(values: Sequence[float], seed: int = SEED) -> tuple[float, np.ndarray]:
    x = np.asarray(values, dtype=float)
    idx = _boot_idx(x.size, np.random.default_rng(seed))
    return float(x.mean()), x[idx].mean(axis=1)


def boot_ratio(
    num: Mapping[str, float], den: Mapping[str, float], seed: int = SEED
) -> tuple[float, np.ndarray, int]:
    """Paired ratio of means over shared items: mean(num)/mean(den)."""
    items = sorted(set(num) & set(den))
    if len(items) < 2:
        return float("nan"), np.array([]), len(items)
    a = np.array([num[i] for i in items])
    b = np.array([den[i] for i in items])
    idx = _boot_idx(len(items), np.random.default_rng(seed))
    boots = a[idx].mean(axis=1) / b[idx].mean(axis=1)
    return float(a.mean() / b.mean()), boots, len(items)


def ci(boots: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    if boots.size == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return float(lo), float(hi)


def holm_reject(pvalues: Mapping[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    names = sorted(pvalues, key=lambda k: pvalues[k])
    m = len(names)
    out = dict.fromkeys(pvalues, False)
    for rank, k in enumerate(names):
        if pvalues[k] <= alpha / (m - rank):
            out[k] = True
        else:
            break
    return out


def overall(per_lang: Mapping[str, str]) -> str:
    vals = list(per_lang.values())
    if sum(v == "confirmed" for v in vals) >= 3:
        return "confirmed"
    if sum(v == "falsified" for v in vals) >= 3:
        return "falsified"
    return "inconclusive"


def combine_pairs(verdicts: Sequence[str]) -> str:
    if verdicts and all(v == "confirmed" for v in verdicts):
        return "confirmed"
    if verdicts and all(v == "falsified" for v in verdicts):
        return "falsified"
    return "inconclusive" if verdicts else "not tested"


def _pairs(keys: Iterable[Key], bases: set[str], engine: str = "vllm") -> list[tuple[str, str]]:
    """(arm, model) pairs present for the given arm bases; one variant per (base, model)."""
    found: dict[tuple[str, str], set[str]] = defaultdict(set)
    for arm, model, eng, _lang, _c in keys:
        if eng == engine and arm_base(arm) in bases:
            found[(arm_base(arm), model)].add(arm)
    pairs = []
    for (base, model), arms in sorted(found.items()):
        if len(arms) > 1:
            raise ValueError(f"several variants of {base} on {model}: {sorted(arms)}; analyse one")
        pairs.append((next(iter(arms)), model))
    return pairs


# --------------------------------------------------------------------------- hypotheses


def h1(tau: Mapping[Key, dict[str, float]], conc: int = 1) -> dict[str, Any]:
    res: dict[str, Any] = {"pairs": {}}
    pair_verdicts = []
    for arm, model in _pairs(tau, {"A3", "A4"}):
        en = tau.get((arm, model, "vllm", "en", conc), {})
        if not en:
            continue
        tau_en = float(np.mean(list(en.values())))
        stats: dict[str, dict[str, Any]] = {}
        p_conf: dict[str, float] = {}
        p_fals: dict[str, float] = {}
        for lang in DRAVIDIAN:
            vals = list(tau.get((arm, model, "vllm", lang, conc), {}).values())
            if len(vals) < 2:
                continue
            m, boots = boot_mean(vals)
            lo, hi = ci(boots)
            p_conf[lang] = float(np.mean(boots >= 1.5))
            p_fals[lang] = float(np.mean(boots < 0.9 * tau_en))
            stats[lang] = {"tau": m, "ci": [lo, hi], "n": len(vals)}
        hc, hf = holm_reject(p_conf), holm_reject(p_fals)
        per_lang = {}
        for lang, s in stats.items():
            if s["ci"][1] < 1.5 and hc[lang]:
                per_lang[lang] = "confirmed"
            elif s["ci"][0] >= 0.9 * tau_en and hf[lang]:
                per_lang[lang] = "falsified"
            else:
                per_lang[lang] = "inconclusive"
        v = overall(per_lang)
        pair_verdicts.append(v)
        res["pairs"][f"{arm}@{model}"] = {
            "tau_en": tau_en,
            "languages": stats,
            "p_confirm": p_conf,
            "p_falsify": p_fals,
            "per_language": per_lang,
            "verdict": v,
        }
    res["verdict"] = combine_pairs(pair_verdicts)
    return res


def ratio_test(
    vals: Mapping[Key, dict[str, float]], arm: str, model: str, conc: int, threshold: float
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    """R_L stats per Dravidian language plus one-sided p-values for R >= thr and R < thr."""
    en = vals.get((arm, model, "vllm", "en", conc), {})
    stats, p_ge, p_lt = {}, {}, {}
    for lang in DRAVIDIAN:
        num = vals.get((arm, model, "vllm", lang, conc), {})
        r, boots, n = boot_ratio(num, en)
        if n < 2:
            continue
        lo, hi = ci(boots)
        stats[lang] = {"R": r, "ci": [lo, hi], "n_items": n}
        p_ge[lang] = float(np.mean(boots < threshold))  # wrong side for "R >= thr"
        p_lt[lang] = float(np.mean(boots >= threshold))  # wrong side for "R < thr"
    return stats, p_ge, p_lt


def h2(tau: Mapping[Key, dict[str, float]], conc: int = 1) -> dict[str, Any]:
    res: dict[str, Any] = {"pairs": {}}
    pair_verdicts = []
    for arm, model in _pairs(tau, {"A5", "A1"}):
        stats, p_ge, p_lt = ratio_test(tau, arm, model, conc, 0.75)
        if not stats:
            continue
        hge, hlt = holm_reject(p_ge), holm_reject(p_lt)
        per_lang = {}
        for lang, s in stats.items():
            if s["ci"][0] >= 0.75 and hge[lang]:
                per_lang[lang] = "confirmed"
            elif s["ci"][1] < 0.75 and hlt[lang]:
                per_lang[lang] = "falsified"
            else:
                per_lang[lang] = "inconclusive"
        v = overall(per_lang)
        pair_verdicts.append(v)
        res["pairs"][f"{arm}@{model}"] = {
            "languages": stats,
            "per_language": per_lang,
            "verdict": v,
        }
    res["verdict"] = combine_pairs(pair_verdicts)
    return res


def speedups(
    tps: Mapping[Key, dict[str, float]], arm: str, model: str, engine: str, lang: str, conc: int
) -> tuple[float, np.ndarray, int]:
    base = next(
        (
            v
            for (a, m, e, la, c), v in tps.items()
            if arm_base(a) == "A0" and m == model and e == engine and la == lang and c == conc
        ),
        {},
    )
    return boot_ratio(tps.get((arm, model, engine, lang, conc), {}), base)


def h4(tps: Mapping[Key, dict[str, float]]) -> dict[str, Any]:
    res: dict[str, Any] = {"pairs": {}}
    pair_verdicts = []
    q = 1 - ALPHA / len(DRAVIDIAN)  # one-sided Bonferroni upper bound
    for arm, model in _pairs(tps, {"A1", "A5"}):
        grid = sorted({k[4] for k in tps if k[0] == arm and k[1] == model and k[2] == "vllm"})
        if len(grid) < 2:
            continue
        cstar: dict[str, int | str] = {}
        idx: dict[str, int] = {}
        for lang in ("en", *DRAVIDIAN):
            pos = len(grid)
            for i, c in enumerate(grid):
                _, boots, n = speedups(tps, arm, model, "vllm", lang, c)
                if n >= 2 and float(np.quantile(boots, q)) < 1.0:
                    pos = i
                    break
            idx[lang] = pos
            cstar[lang] = grid[pos] if pos < len(grid) else f">{grid[-1]}"
        per_lang = {}
        for lang in DRAVIDIAN:
            if "en" not in idx or lang not in idx:
                continue
            d = abs(idx[lang] - idx["en"])
            per_lang[lang] = "confirmed" if d >= 2 else "falsified" if d <= 1 else "inconclusive"
        v = overall(per_lang)
        pair_verdicts.append(v)
        res["pairs"][f"{arm}@{model}"] = {
            "grid": grid,
            "crossover": cstar,
            "per_language": per_lang,
            "verdict": v,
        }
    res["verdict"] = combine_pairs(pair_verdicts)
    return res


# --------------------------------------------------------------------------- positions (H3, G2)


@dataclass
class PositionData:
    n_requests: int
    n_aligned: int
    # per request: list of (token_id, accepted)
    per_request: list[tuple[str, list[tuple[int, bool]]]]  # (language, positions)


def positions_for(rows: Iterable[Mapping[str, Any]]) -> PositionData:
    out: list[tuple[str, list[tuple[int, bool]]]] = []
    n = aligned = 0
    for r in rows:
        raw = r.get("spec_engine_raw") or {}
        ids = r.get("output_token_ids")
        if raw.get("per_step_accepted") is None or ids is None:
            n += 1
            continue
        n += 1
        try:
            ev = evaluated_positions(ids, raw["per_step_drafted"], raw["per_step_accepted"])
        except (AlignmentError, KeyError, TypeError):
            continue
        aligned += 1
        out.append((r["language"], [(e.token_id, e.accepted) for e in ev]))
    return PositionData(n, aligned, out)


def _cluster_rate(
    reqs: Sequence[list[tuple[int, bool]]], select: Callable[[int], bool]
) -> tuple[float, np.ndarray]:
    """Acceptance rate over selected positions; cluster bootstrap over requests."""
    acc = np.array([sum(a for t, a in ps if select(t)) for ps in reqs], dtype=float)
    tot = np.array([sum(1 for t, _ in ps if select(t)) for ps in reqs], dtype=float)
    total = float(tot.sum())
    if total == 0:
        return float("nan"), np.array([])
    idx = _boot_idx(len(reqs), np.random.default_rng(SEED))
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = acc[idx].sum(axis=1) / tot[idx].sum(axis=1)
    return float(acc.sum()) / total, boots[~np.isnan(boots)]


def h3(
    rows: Sequence[Mapping[str, Any]], token_class: Mapping[int, str], conc: int = 1
) -> dict[str, Any]:
    res: dict[str, Any] = {"pairs": {}}
    pair_verdicts = []
    keys = {(r["arm"], r["model"], r["engine"], r["language"], int(r["concurrency"])) for r in rows}
    for arm, model in _pairs(keys, {"A1", "A5"}):
        sel = [
            r
            for r in rows
            if r["arm"] == arm
            and r["model"] == model
            and r["engine"] == "vllm"
            and int(r["concurrency"]) == conc
        ]
        pdata = positions_for(sel)
        frac = pdata.n_aligned / pdata.n_requests if pdata.n_requests else 0.0
        entry: dict[str, Any] = {"aligned_fraction": frac, "n_requests": pdata.n_requests}
        if frac < 0.99:
            entry["verdict"] = "not testable"
            res["pairs"][f"{arm}@{model}"] = entry
            continue
        en = [ps for lang, ps in pdata.per_request if lang == "en"]
        dr = [ps for lang, ps in pdata.per_request if lang in DRAVIDIAN]
        r_en, b_en = _cluster_rate(en, lambda t: True)
        r_ct, b_ct = _cluster_rate(dr, lambda t: token_class.get(t) == "BYTE_FRAGMENT_CONT")
        if b_en.size == 0 or b_ct.size == 0:
            entry["verdict"] = "not testable"
            res["pairs"][f"{arm}@{model}"] = entry
            continue
        k = min(b_en.size, b_ct.size)
        diff = b_ct[:k] - b_en[:k]
        lo, hi = ci(diff)
        v = "confirmed" if lo >= -0.05 else "falsified" if hi < -0.05 else "inconclusive"
        entry.update(rate_cont_dravidian=r_ct, rate_en_all=r_en, diff_ci=[lo, hi], verdict=v)
        pair_verdicts.append(v)
        res["pairs"][f"{arm}@{model}"] = entry
    res["verdict"] = combine_pairs(pair_verdicts)
    return res


def g2(
    rows: Sequence[Mapping[str, Any]],
    token_class: Mapping[int, str],
    draft_vocabs: Mapping[str, set[int]] | None = None,
    conc: int = 1,
) -> dict[str, Any]:
    res: dict[str, Any] = {"pairs": {}, "coverage_split": {}}
    ok_pairs = []
    keys = {(r["arm"], r["model"], r["engine"], r["language"], int(r["concurrency"])) for r in rows}
    for arm, model in _pairs(keys, {"A1", "A5"}):
        sel = [
            r
            for r in rows
            if r["arm"] == arm
            and r["model"] == model
            and r["engine"] == "vllm"
            and int(r["concurrency"]) == conc
        ]
        pd_ = positions_for(sel)
        en = [ps for lang, ps in pd_.per_request if lang == "en"]
        dr = [ps for lang, ps in pd_.per_request if lang in DRAVIDIAN]
        _, b_lead = _cluster_rate(dr, lambda t: token_class.get(t) == "BYTE_FRAGMENT_LEAD")
        _, b_cont = _cluster_rate(dr, lambda t: token_class.get(t) == "BYTE_FRAGMENT_CONT")
        k = min(b_lead.size, b_cont.size)
        a_lo, a_hi = ci(b_lead[:k] - b_cont[:k]) if k else (float("nan"), float("nan"))
        cond_a = bool(k) and (a_lo > 0 or a_hi < 0)
        # (b) class standardisation, bootstrap over requests jointly
        classes = sorted({token_class.get(t, "UNKNOWN") for ps in en for t, _ in ps})
        rng = np.random.default_rng(SEED)
        ratios = []
        point = _standardised_ratio(en, dr, token_class, classes)
        for _ in range(min(N_BOOT, 2000)):  # nested resampling: capped for runtime
            ie = rng.integers(0, len(en), len(en)) if en else []
            idr = rng.integers(0, len(dr), len(dr)) if dr else []
            rr = _standardised_ratio(
                [en[i] for i in ie], [dr[i] for i in idr], token_class, classes
            )
            if rr is not None:
                ratios.append(rr)
        b_lo, b_hi = ci(np.array(ratios)) if ratios else (float("nan"), float("nan"))
        cond_b = point is not None and bool(ratios) and b_hi <= 0.5
        res["pairs"][f"{arm}@{model}"] = {
            "lead_minus_cont_ci": [a_lo, a_hi],
            "a": cond_a,
            "standardised_over_raw_gap": point,
            "ratio_ci": [b_lo, b_hi],
            "b": cond_b,
        }
        ok_pairs.append(cond_a and cond_b)
    for arm, model in _pairs(keys, {"A3", "A4"}):
        vocab = (draft_vocabs or {}).get(arm_base(arm))
        if vocab is None:
            continue
        sel = [
            r
            for r in rows
            if r["arm"] == arm and r["model"] == model and int(r["concurrency"]) == conc
        ]
        pd_ = positions_for(sel)
        per_lang = {}
        for lang in LANGS:
            reqs = [ps for la, ps in pd_.per_request if la == lang]
            rate, boots = _cluster_rate(reqs, vocab.__contains__)
            per_lang[lang] = {"in_vocab_acceptance": rate, "ci": list(ci(boots))}
        res["coverage_split"][f"{arm}@{model}"] = per_lang
    res["gate"] = "YES" if ok_pairs and all(ok_pairs) else "NO"
    return res


def _standardised_ratio(
    en: Sequence[list[tuple[int, bool]]],
    dr: Sequence[list[tuple[int, bool]]],
    token_class: Mapping[int, str],
    classes: Sequence[str],
) -> float | None:
    def rates(
        reqs: Sequence[list[tuple[int, bool]]],
    ) -> tuple[dict[str, float], dict[str, int], float]:
        a: dict[str, int] = defaultdict(int)
        n: dict[str, int] = defaultdict(int)
        for ps in reqs:
            for t, ok in ps:
                c = token_class.get(t, "UNKNOWN")
                a[c] += int(ok)
                n[c] += 1
        tot = sum(n.values())
        overall_rate = sum(a.values()) / tot if tot else float("nan")
        return {c: a[c] / n[c] for c in n if n[c]}, dict(n), overall_rate

    _r_en, n_en, all_en = rates(en)
    r_dr, _, all_dr = rates(dr)
    raw_gap = all_en - all_dr
    if not raw_gap > 0:
        return None
    shared = [c for c in classes if c in r_dr and n_en.get(c)]
    w = sum(n_en[c] for c in shared)
    if not w:
        return None
    std_dr = sum(n_en[c] / w * r_dr[c] for c in shared)
    return (all_en - std_dr) / raw_gap


# --------------------------------------------------------------------------- gates


def g1(tau: Mapping[Key, dict[str, float]], conc: int = 1) -> dict[str, Any]:
    res: dict[str, Any] = {"pairs": {}}
    yes = False
    for arm, model in _pairs(tau, {"A1", "A3", "A4", "A5"}):
        g_stats, _, p_gap = ratio_test(tau, arm, model, conc, 0.75)  # wrong side: R >= 0.75
        a_stats, p_adv, _ = ratio_test(tau, arm, model, conc, 1.25)  # wrong side: R < 1.25
        hg, ha = holm_reject(p_gap), holm_reject(p_adv)
        gap_langs = [la for la, s in g_stats.items() if s["ci"][1] < 0.75 and hg[la]]
        adv_langs = [la for la, s in a_stats.items() if s["ci"][0] > 1.25 and ha[la]]
        hit = len(gap_langs) >= 2 or len(adv_langs) >= 2
        yes = yes or hit
        res["pairs"][f"{arm}@{model}"] = {
            "R": {la: s["R"] for la, s in g_stats.items()},
            "gap_languages": gap_langs,
            "advantage_languages": adv_langs,
            "hit": hit,
        }
    res["gate"] = "YES" if yes else "NO"
    return res


def g3(speedup_ratio_ci_lo: Mapping[str, float]) -> str:
    """PREREG G3: new drafter / strongest public arm; paired speed-up lower CI > 1.05 in >= 3/4."""
    wins = [la for la in DRAVIDIAN if speedup_ratio_ci_lo.get(la, float("-inf")) > 1.05]
    return "YES" if len(wins) >= 3 else "NO"


# --------------------------------------------------------------------------- tables / main


def stats_table(
    tau: Mapping[Key, dict[str, float]], tps: Mapping[Key, dict[str, float]]
) -> list[dict[str, Any]]:
    rows = []
    for k in sorted(set(tau) | set(tps)):
        arm, model, engine, lang, conc = k
        t = list(tau.get(k, {}).values())
        row: dict[str, Any] = {
            "arm": arm,
            "model": model,
            "engine": engine,
            "language": lang,
            "concurrency": conc,
            "n_items": len(tps.get(k, {})),
        }
        if len(t) >= 2:
            m, boots = boot_mean(t)
            row.update(tau=m, tau_ci_lo=ci(boots)[0], tau_ci_hi=ci(boots)[1])
        if arm_base(arm) != "A0":
            s, boots, n = speedups(tps, arm, model, engine, lang, conc)
            if n >= 2:
                lo, hi = ci(boots)
                row.update(speedup=s, speedup_ci_lo=lo, speedup_ci_hi=hi)
        rows.append(row)
    return rows


def run_all(
    d: Data,
    token_class: Mapping[int, str] | None = None,
    draft_vocabs: Mapping[str, set[int]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept, excluded = apply_exclusions(d)
    tau = per_prompt(kept, request_tau)
    tps = per_prompt(kept, request_tps)
    verdicts: dict[str, Any] = {
        "prereg_commit": "6023a78",
        "interpretations": INTERPRETATIONS,
        "excluded_cells": excluded,
        "n_requests_kept": len(kept),
        "H1": h1(tau),
        "H2": h2(tau),
        "H4": h4(tps),
        "G1": g1(tau),
        "G3": "not evaluated (no new drafter yet)",
    }
    if token_class is not None:
        verdicts["H3"] = h3(kept, token_class)
        verdicts["G2"] = g2(kept, token_class, draft_vocabs)
    else:
        verdicts["H3"] = verdicts["G2"] = "token classes not loaded"
    return stats_table(tau, tps), verdicts


def _load_token_classes() -> dict[int, str]:
    import yaml

    from indicspec.tokens.vocab import PinnedTokenizer, classify_vocab, fetch_tokenizer_json

    cfg = yaml.safe_load((REPO / "configs/models.yaml").read_text(encoding="utf-8"))
    t = cfg["targets"]["qwen3-8b"]
    info = classify_vocab(fetch_tokenizer_json(PinnedTokenizer(t["repo"], t["revision"])))
    return {i: str(v.cls) for i, v in info.items()}


def _load_draft_vocabs() -> dict[str, set[int]]:
    import yaml

    from indicspec.hub import local_snapshot
    from indicspec.tokens.draftvocab import load_draft_vocab

    arms = yaml.safe_load((REPO / "configs/arms.yaml").read_text(encoding="utf-8"))
    out = {}
    for key in ("A3", "A4"):
        dcfg = arms["drafters"][key]
        try:
            out[key] = load_draft_vocab(local_snapshot(dcfg["repo"], dcfg["revision"])).target_ids
        except Exception:  # not cached on this machine -> coverage split skipped, reported
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="results/raw/<run> (repeatable)")
    ap.add_argument("--name", default=None, help="output name (default: runs joined by '+')")
    ap.add_argument("--telemetry-utc-offset", type=float, default=None)
    ap.add_argument("--no-positions", action="store_true", help="skip H3/G2 (no tokenizer)")
    a = ap.parse_args(argv)
    d = load([REPO / "results/raw" / r for r in a.run], a.telemetry_utc_offset)
    tc = None if a.no_positions else _load_token_classes()
    dv = None if a.no_positions else _load_draft_vocabs()
    table, verdicts = run_all(d, tc, dv)
    name = a.name or "+".join(a.run)
    out = REPO / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in table:
        keys += [k for k in r if k not in keys]
    with open(out / f"{name}_confirm.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in table:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    (out / f"{name}_verdicts.json").write_text(
        json.dumps(verdicts, indent=2, default=float), encoding="utf-8"
    )
    summary = {
        k: (v.get("verdict") or v.get("gate")) if isinstance(v, dict) else v
        for k, v in verdicts.items()
        if k in ("H1", "H2", "H3", "H4", "G1", "G2", "G3")
    }
    print(json.dumps({"excluded_cells": len(verdicts["excluded_cells"]), **summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
