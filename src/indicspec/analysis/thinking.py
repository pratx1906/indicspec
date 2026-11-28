"""Sequential thinking-mode study (exploratory, D-028): analysis, convergence and stage planning.

Reads one or more staged thinking-mode runs (Qwen3 thinking on) and the thinking-off confirmatory
run on the same items, and reports per (arm, language, concurrency):

* whole-output speed-up vs no speculation (thinking on), paired over items, with drift (stability);
* acceptance length tau for the whole output, the reasoning part (up to and including ``</think>``)
  and the answer part, from vLLM's per-step accepted counts;
* how much of the output is reasoning, how often thinking finished within the token limit, and the
  share of reasoning letters in the prompt's language (vs Latin: does Qwen3 reason in English?);
* the on/off ratio: speed-up with thinking on divided by speed-up with thinking off on the same
  items (bootstrap over items, both runs resampled together).

It also bootstraps the Dravidian/English speed-up ratio (on, off, and on/off) and, with
``--plan``, writes the next stage's config: the next items, only for concurrencies with a cell
whose speed-up CI half-width exceeds ``--target`` of the estimate (or that drifts by more), up to
``--max-items`` items per language in total.

Usage (WSL, venv-cpu; needs the pinned Qwen3 tokenizer in the HF cache):
  python -m indicspec.analysis.thinking --runs explore_thinking_s1_vllm_4b [...] \
      [--plan configs/explore_thinking_s2_vllm_4b.yaml]
Outputs: results/tables/explore_thinking_vllm_4b.csv, ..._contrasts.csv, printed summary.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from indicspec.analysis.convergence import boot_ratio, drift, needed
from indicspec.rawio import read_jsonl

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "results/raw"
TABLES = REPO / "results/tables"
THINK_END = 151668  # Qwen3 </think>
DRAV = ("ta", "te", "kn", "ml")
SCRIPT = {
    "en": (0x0041, 0x007A),
    "hi": (0x0900, 0x097F),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
}


def segments(r: dict[str, Any]) -> dict[str, float]:
    """Token and verify-step counts for the reasoning and answer parts of one request."""
    ids = r.get("output_token_ids") or []
    n = int(r.get("output_tokens") or len(ids))
    end = ids.index(THINK_END) + 1 if THINK_END in ids else n  # unfinished thinking: all reasoning
    out = {
        "tok_r": float(end),
        "tok_a": float(n - end),
        "finished": float(THINK_END in ids),
        "steps_r": 0.0,
        "steps_a": 0.0,
    }
    raw = r.get("spec_engine_raw") or {}
    acc = raw.get("per_step_accepted")
    if acc:
        pos = n - sum(a + 1 for a in acc)  # tokens emitted before the first verify step
        for a in acc:
            out["steps_r" if pos < end else "steps_a"] += 1
            pos += a + 1
    return out


def script_share(text: str, lang: str) -> float | None:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    lo, hi = SCRIPT[lang]
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters)


def _tps(r: dict[str, Any]) -> float | None:
    e2e = r.get("e2e_latency_s")
    if not e2e or r.get("error") or not r.get("output_tokens"):
        return None
    return float(r["output_tokens"]) / float(e2e)


def load(runs: list[str]) -> dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]]:
    """(arm, lang, conc) -> prompt_id -> list of request rows (over runs/repeats)."""
    out: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in runs:
        for d in sorted({f.parent for f in (RAW / run).glob("*/requests.jsonl*")}):
            for r in read_jsonl(d / "requests.jsonl"):
                if _tps(r) is not None:
                    out[(r["arm"], r["language"], int(r["concurrency"]))][r["prompt_id"]].append(r)
    return out


def _onoff(m: np.ndarray) -> float:
    """Columns: on-arm, on-A0, off-arm, off-A0 tokens/s per item -> speed-up on / speed-up off."""
    on = float(m[:, 0].sum()) / float(m[:, 1].sum())
    return on / (float(m[:, 2].sum()) / float(m[:, 3].sum()))


def _mean_tps(rs: list[dict[str, Any]]) -> float:
    return float(np.mean([_tps(r) for r in rs]))


def analyse(
    runs: list[str], ref: str, decode: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    on, off = load(runs), load([ref])
    rows: list[dict[str, Any]] = []
    ratio_items: dict[tuple[str, int], dict[str, dict[str, tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (arm, lang, conc), g in sorted(on.items()):
        segs = {pid: [segments(r) for r in rs] for pid, rs in g.items()}
        row: dict[str, Any] = {"arm": arm, "language": lang, "concurrency": conc, "n_items": len(g)}
        row["reasoning_share"] = float(
            np.mean(
                [s["tok_r"] / max(s["tok_r"] + s["tok_a"], 1) for v in segs.values() for s in v]
            )
        )
        row["thinking_finished"] = float(np.mean([s["finished"] for v in segs.values() for s in v]))
        if decode is not None and arm == "A0":
            shares = []
            for rs in g.values():
                for r in rs:
                    ids = r.get("output_token_ids") or []
                    end = ids.index(THINK_END) if THINK_END in ids else len(ids)
                    s = script_share(decode(ids[:end]), lang)
                    if s is not None:
                        shares.append(s)
            row["reasoning_in_language"] = float(np.mean(shares)) if shares else None
        if arm != "A0":
            base = on.get(("A0", lang, conc), {})
            ids = sorted(set(g) & set(base))
            if len(ids) >= 3:
                a = [_mean_tps(g[i]) for i in ids]
                b = [_mean_tps(base[i]) for i in ids]
                est, lo, hi = boot_ratio(a, b)
                row.update(speedup=est, ci_lo=lo, ci_hi=hi, rel_half_width=(hi - lo) / 2 / est)
                row["drift"] = drift(a, b)
                for part, tk, sk in (
                    ("all", None, None),
                    ("reasoning", "tok_r", "steps_r"),
                    ("answer", "tok_a", "steps_a"),
                ):
                    if part == "all":
                        num = [sum(s["tok_r"] + s["tok_a"] for s in segs[i]) for i in ids]
                        den = [sum(s["steps_r"] + s["steps_a"] for s in segs[i]) for i in ids]
                    else:
                        assert tk is not None and sk is not None
                        num = [float(sum(sg[tk] for sg in segs[i])) for i in ids]
                        den = [float(sum(sg[sk] for sg in segs[i])) for i in ids]
                    keep = [(x, y) for x, y in zip(num, den, strict=True) if y > 0]
                    if len(keep) >= 3:
                        t, tlo, thi = boot_ratio([k[0] for k in keep], [k[1] for k in keep])
                        row.update({f"tau_{part}": t, f"tau_{part}_lo": tlo, f"tau_{part}_hi": thi})
                ref_a, ref_b = off.get((arm, lang, conc), {}), off.get(("A0", lang, conc), {})
                both = [i for i in ids if i in ref_a and i in ref_b]
                if len(both) >= 3:
                    x = np.array(
                        [
                            [
                                _mean_tps(g[i]),
                                _mean_tps(base[i]),
                                _mean_tps(ref_a[i]),
                                _mean_tps(ref_b[i]),
                            ]
                            for i in both
                        ]
                    )
                    rng = np.random.default_rng(1234)
                    boots = [_onoff(x[rng.integers(0, len(x), len(x))]) for _ in range(2000)]
                    q = np.quantile(boots, [0.025, 0.975])
                    row.update(
                        speedup_off=float(x[:, 2].sum()) / float(x[:, 3].sum()),
                        on_over_off=_onoff(x),
                        on_over_off_lo=float(q[0]),
                        on_over_off_hi=float(q[1]),
                        n_paired_off=len(both),
                    )
                    for i, v in zip(both, x, strict=True):
                        ratio_items[(arm, conc)][i][lang] = (v[0] / v[1], v[2] / v[3])
        rows.append(row)
    contrasts: list[dict[str, Any]] = []
    for (arm, conc), items in sorted(ratio_items.items()):
        full = [v for v in items.values() if "en" in v and all(d in v for d in DRAV)]
        if len(full) < 3:
            continue
        m = np.array([[[v[x][k] for x in ("en", *DRAV)] for k in (0, 1)] for v in full])  # n,2,5

        def c(mm: np.ndarray) -> tuple[float, float, float]:
            on_c = mm[:, 0, 1:].mean() / mm[:, 0, 0].mean()
            off_c = mm[:, 1, 1:].mean() / mm[:, 1, 0].mean()
            return float(on_c), float(off_c), float(on_c / off_c)

        rng = np.random.default_rng(1234)
        cboots = np.array([c(m[rng.integers(0, len(m), len(m))]) for _ in range(2000)])
        cest = c(m)
        qlo, qhi = np.quantile(cboots, 0.025, axis=0), np.quantile(cboots, 0.975, axis=0)
        contrasts.append(
            {
                "arm": arm,
                "concurrency": conc,
                "n_items": len(full),
                "drav_over_en_on": cest[0],
                "on_lo": float(qlo[0]),
                "on_hi": float(qhi[0]),
                "drav_over_en_off": cest[1],
                "off_lo": float(qlo[1]),
                "off_hi": float(qhi[1]),
                "on_over_off": cest[2],
                "ratio_lo": float(qlo[2]),
                "ratio_hi": float(qhi[2]),
            }
        )
    # Convergence is judged on the quantity the study compares: the paired on/off ratio (item
    # heterogeneity cancels in the pairing). Without a thinking-off reference, the speed-up itself.
    status: dict[tuple[str, int], dict[str, Any]] = {}
    for r in rows:
        if "speedup" in r:
            if "on_over_off" in r:
                prec = (r["on_over_off_hi"] - r["on_over_off_lo"]) / 2 / r["on_over_off"]
            else:
                prec = r["rel_half_width"]
            r["converged"] = prec <= TARGET[0] and r["drift"] <= TARGET[0]
            key = (r["language"], r["concurrency"])
            grp = status.setdefault(key, {"n": r["n_items"], "need": 0})
            grp["need"] = max(grp["need"], needed(r["n_items"], max(prec, r["drift"]), TARGET[0]))
    return rows, contrasts, status


TARGET = [0.05]


def plan(
    status: dict[tuple[str, int], dict[str, Any]], s1: Path, dest: Path, max_items: int
) -> None:
    cfg = yaml.safe_load(s1.read_text(encoding="utf-8"))
    have = max(s["n"] for s in status.values())
    concs = sorted({c for (_, c), s in status.items() if s["need"] > s["n"]})
    want = min(max_items, max((s["need"] for s in status.values()), default=have))
    if not concs or want <= have:
        print("plan: every cell converged or the item cap is reached; no further stage")
        return
    cfg.update(
        run_name=dest.stem,
        prompt_offset=have + int(cfg.get("prompt_offset", 0)),
        max_prompts_per_cell=want - have,
        concurrency=concs,
    )
    head = (
        "# Stage written by indicspec.analysis.thinking --plan (D-028): items "
        f"{have + 1}-{want}\n# of the pooled order, concurrencies with an unconverged cell: "
        f"{concs}.\n"
    )
    dest.write_text(head + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"plan: wrote {dest.relative_to(REPO)} (+{want - have} items, concurrency {concs})")


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref", default="confirm_vllm_4b")
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--max-items", type=int, default=40)
    ap.add_argument("--plan", type=Path)
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--out", default="explore_thinking_vllm_4b", help="table name prefix")
    a = ap.parse_args(argv)
    TARGET[0] = a.target
    decode: Any = None
    if not a.no_decode:
        from indicspec.tokens.vocab import PinnedTokenizer, load_tokenizer

        m = yaml.safe_load((REPO / "configs/models.yaml").read_text(encoding="utf-8"))
        t = m["targets"]["qwen3-4b"]
        tok = load_tokenizer(PinnedTokenizer(t["repo"], t["revision"]))

        def decode(ids: list[int]) -> str:
            return str(tok.decode(ids, skip_special_tokens=True))

    rows, contrasts, status = analyse(a.runs, a.ref, decode)
    _write(TABLES / f"{a.out}.csv", rows)
    if contrasts:
        _write(TABLES / f"{a.out}_contrasts.csv", contrasts)
    print(
        "arm    lang c  n  speedup [CI]        off   on/off [CI]         tau_r  tau_a  reas%  fin"
    )
    for r in rows:
        if "speedup" not in r:
            if r["arm"] == "A0":
                print(
                    f"A0     {r['language']:4} {r['concurrency']:<2} {r['n_items']:<2} "
                    f"reasoning {r['reasoning_share']:.0%}, finished {r['thinking_finished']:.0%}, "
                    f"in-language {r.get('reasoning_in_language') or float('nan'):.0%}"
                )
            continue
        print(
            f"{r['arm']:6} {r['language']:4} {r['concurrency']:<2} {r['n_items']:<2} "
            f"{r['speedup']:.2f} [{r['ci_lo']:.2f},{r['ci_hi']:.2f}]  "
            f"{r.get('speedup_off', math.nan):.2f}  {r.get('on_over_off', math.nan):.2f} "
            f"[{r.get('on_over_off_lo', math.nan):.2f},{r.get('on_over_off_hi', math.nan):.2f}]  "
            f"{r.get('tau_reasoning', math.nan):5.2f}  {r.get('tau_answer', math.nan):5.2f}  "
            f"{r['reasoning_share']:4.0%}  {r['thinking_finished']:3.0%}"
        )
    for c in contrasts:
        print(
            f"contrast {c['arm']} c{c['concurrency']} n={c['n_items']}: Dravidian/English on "
            f"{c['drav_over_en_on']:.2f} [{c['on_lo']:.2f},{c['on_hi']:.2f}], off "
            f"{c['drav_over_en_off']:.2f} [{c['off_lo']:.2f},{c['off_hi']:.2f}], on/off "
            f"{c['on_over_off']:.2f} [{c['ratio_lo']:.2f},{c['ratio_hi']:.2f}]"
        )
    unconv = {k: v for k, v in status.items() if v["need"] > v["n"]}
    print(
        f"convergence (target {a.target:.0%}): {len(status) - len(unconv)}/{len(status)} "
        f"(language, concurrency) groups converged; items needed where not: "
        f"{ {f'{k[0]}|c{k[1]}': v['need'] for k, v in sorted(unconv.items())} }"
    )
    if a.plan:
        dest = a.plan if a.plan.is_absolute() else REPO / a.plan
        plan(status, REPO / f"configs/{a.runs[0]}.yaml", dest, a.max_items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
