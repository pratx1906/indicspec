"""Draft-length tuning analysis (PREREG §5) and confirmatory-config generation.

Rule (frozen): on the tuning split, concurrency 1, pick for each arm the draft setting with the
highest geometric-mean speed-up over the six languages; use it for every language; report the
full grid.

Speed-up per (arm setting, language) = mean over items of the arm's per-request tokens/s divided
by the mean over the same items of A0's (paired by prompt id).

Outputs:
  results/tables/<run>_grid.csv     every setting x language speed-up and tau
  results/tables/<run>_choice.csv   the chosen setting per arm
  configs/confirm_<engine>_<model>.yaml (with --write-config) using the chosen settings and the
  frozen confirmatory design (report split, 100 prompts, 3 repeats, PREREG concurrency grid).

Usage: python -m indicspec.analysis.tuning --run tune_vllm_4b [--write-config]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[3]
LANGS = ["en", "hi", "ta", "te", "kn", "ml"]
CONFIRM_CONCURRENCY = {"qwen3-4b": [1, 2, 4, 8, 16, 32], "qwen3-8b": [1, 2, 4, 8]}
CONFIRM_ARMS = {"qwen3-4b": ["A0", "A1", "A5"], "qwen3-8b": ["A0", "A3", "A4", "A5"]}


def _rows(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _tps(r: dict[str, Any]) -> float | None:
    if r.get("error") or not r.get("e2e_latency_s") or not r.get("output_tokens"):
        return None
    return float(r["output_tokens"]) / float(r["e2e_latency_s"])


def grid(run_dir: Path) -> list[dict[str, Any]]:
    """Speed-up and tau per (arm label, language) at concurrency 1."""
    by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for r in _rows(block / "requests.jsonl") if (block / "requests.jsonl").exists() else []:
            if int(r["concurrency"]) == 1 and r["split"] == "tuning":
                by_setting[r["arm"]].append(r)
    base: dict[tuple[str, str], float] = {}
    for r in by_setting.get("A0", []):
        v = _tps(r)
        if v is not None:
            base[(r["language"], r["prompt_id"])] = v
    out = []
    for setting, rs in sorted(by_setting.items()):
        if setting == "A0":
            continue
        for lang in LANGS:
            lr = [r for r in rs if r["language"] == lang]
            pairs = [
                (v, base[(lang, r["prompt_id"])])
                for r in lr
                if (v := _tps(r)) is not None and (lang, r["prompt_id"]) in base
            ]
            a = sum(r.get("spec_accepted_tokens") or 0 for r in lr)
            s = sum(r.get("spec_verify_steps") or 0 for r in lr)
            out.append(
                {
                    "setting": setting,
                    "arm": setting.split(".")[0],
                    "language": lang,
                    "n_pairs": len(pairs),
                    "speedup": (sum(p[0] for p in pairs) / len(pairs))
                    / (sum(p[1] for p in pairs) / len(pairs))
                    if pairs
                    else None,
                    "tau": (a + s) / s if s else None,
                }
            )
    return out


def choose(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per arm, the setting with the highest geometric-mean speed-up over all six languages."""
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["speedup"]:
            by[r["arm"]][r["setting"]].append(float(r["speedup"]))
    best: dict[str, dict[str, Any]] = {}
    for arm, settings in by.items():
        scored = {
            s: math.exp(sum(math.log(v) for v in vs) / len(vs))
            for s, vs in settings.items()
            if len(vs) == len(LANGS)  # a setting missing a language cannot win
        }
        if scored:
            s = max(scored, key=lambda k: scored[k])
            best[arm] = {"setting": s, "geomean_speedup": scored[s], "all": scored}
    return best


def _label_override(run_cfg: dict[str, Any], setting: str) -> dict[str, Any] | None:
    arm, _, label = setting.partition(".")
    for a in run_cfg["arms"]:
        if a["arm"] == arm and a.get("label", "default") == (label or "default"):
            return a.get("spec_override")
    return None


def confirm_config(run_cfg: dict[str, Any], best: dict[str, dict[str, Any]]) -> dict[str, Any]:
    model = run_cfg["model"]
    arms: list[dict[str, Any]] = []
    for arm in CONFIRM_ARMS[model]:
        if arm == "A0":
            arms.append({"arm": "A0"})
            continue
        if arm not in best:
            raise ValueError(f"no tuned setting for {arm}")
        setting = best[arm]["setting"]
        arms.append(
            {
                "arm": arm,
                "label": setting.partition(".")[2] or "default",
                "spec_override": _label_override(run_cfg, setting),
            }
        )
    return {
        "run_name": f"confirm_{run_cfg['engine']}_{model.replace('qwen3-', '')}",
        "engine": run_cfg["engine"],
        "model": model,
        "split": "report",
        "languages": LANGS,
        "tasks": run_cfg["tasks"],
        "arms": arms,
        "concurrency": CONFIRM_CONCURRENCY[model],
        "repeats": 3,
        "load_mode": "closed_loop",
        "max_tokens": 256,
        "ignore_eos": False,
        "max_model_len": 4096,
        "max_prompts_per_cell": 100,
        "temperature": 0.0,
        "seed": 1234,
        "min_enforced_power_w": 140,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--write-config", action="store_true")
    a = ap.parse_args(argv)
    run_dir = REPO / "results/raw" / a.run
    rows = grid(run_dir)
    out = REPO / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{a.run}_grid.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["setting", "arm", "language", "n_pairs", "speedup", "tau"]
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    best = choose(rows)
    with open(out / f"{a.run}_choice.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh)
        cw.writerow(["arm", "chosen_setting", "geomean_speedup", "all_settings"])
        for arm, b in sorted(best.items()):
            cw.writerow(
                [
                    arm,
                    b["setting"],
                    f"{b['geomean_speedup']:.4f}",
                    json.dumps({k: round(v, 4) for k, v in b["all"].items()}),
                ]
            )
    print(json.dumps({k: (v["setting"], round(v["geomean_speedup"], 3)) for k, v in best.items()}))
    if a.write_config:
        run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        cfg = confirm_config(run_cfg, best)
        dst = REPO / "configs" / f"{cfg['run_name']}.yaml"
        header = (
            f"# Confirmatory sweep (PREREG, frozen at 6023a78). Draft settings chosen by "
            f"indicspec.analysis.tuning from {a.run}.\n"
        )
        dst.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"wrote {dst.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
