"""Sweep runner (WS4; `make sweep CONFIG=...`). Linux/WSL only.

One server launch per (engine, arm, draft-params) "block". Inside a block, every
(language, concurrency, repeat) cell runs on the same server, in a seeded shuffled order
(block randomisation, spec WS4.4), so thermal drift cannot align with one factor.

Per cell we write ``requests.jsonl`` rows (schema request_log.v1) and one line in ``cells.jsonl``
with wall-clock start/end, wall time, GPU energy (NVML total-energy counter delta, C30), and the
prompt count. Telemetry (clocks/power/temperature/memory, 1 Hz) is sampled per block; thermal and
VRAM rules are evaluated per cell from timestamps at analysis time. Cells already present in
``cells.jsonl`` are skipped, so an interrupted sweep resumes.

Config (YAML):
  run_name: str
  engine: vllm | sglang
  model: qwen3-4b | qwen3-8b
  split: pilot | tuning | report
  languages: [en, hi, ...]
  tasks: [chat, qa, ...]
  arms: [{arm: A0}, {arm: A5, spec_override: {num_speculative_tokens: 7}, label: k7}, ...]
  concurrency: [1, 4, ...]
  repeats: 3
  load_mode: closed_loop
  max_tokens: 256
  ignore_eos: false
  max_model_len: 4096
  max_prompts_per_cell: 30
  seed: 1234
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from indicspec.env_probe import probe_gpu
from indicspec.harness import gpu_guard
from indicspec.harness.loadgen import LoadSpec, Prompt, run_load, to_log_row
from indicspec.harness.manifest import build_manifest, now_utc
from indicspec.harness.schema import validate
from indicspec.harness.server import REPO, Server, build_plan, load_configs
from indicspec.harness.telemetry import TelemetrySampler
from indicspec.hub import local_snapshot
from indicspec.rawio import open_append


@dataclass(frozen=True)
class Cell:
    language: str
    concurrency: int
    repeat: int

    @property
    def key(self) -> str:
        return f"{self.language}|c{self.concurrency}|r{self.repeat}"


def load_prompts(
    bench_dir: Path,
    split: str,
    languages: Iterable[str],
    tasks: Iterable[str],
    max_prompt_tokens: int,
    max_per_cell: int,
    seed: int,
) -> tuple[dict[str, list[Prompt]], dict[str, int]]:
    """Prompts per language (pre-tokenised ids from the frozen bench), with exclusion counts.

    The same item ids are kept across languages where possible (parallel content): items are
    chosen by sorted item_id after a seeded shuffle that is identical for every language.
    """
    langs, tsk = set(languages), set(tasks)
    rows: list[dict[str, Any]] = []
    for f in sorted(bench_dir.glob("*.jsonl")):
        if f.stem not in tsk:
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["split"] == split and r["lang"] in langs:
                    rows.append(r)
    excluded: dict[str, int] = {}
    by_lang: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        key = f"{r['task']}:{r['item_id']}"
        if len(r["prompt_token_ids"]) > max_prompt_tokens:
            excluded[r["lang"]] = excluded.get(r["lang"], 0) + 1
            continue
        by_lang.setdefault(r["lang"], {})[key] = r
    common = sorted(set.intersection(*(set(v) for v in by_lang.values()))) if by_lang else []
    random.Random(seed).shuffle(common)
    chosen = common[:max_per_cell]
    out = {
        lang: [
            Prompt(
                f"{by_lang[lang][k]['task']}:{by_lang[lang][k]['item_id']}",
                lang,
                by_lang[lang][k]["task"],
                [int(t) for t in by_lang[lang][k]["prompt_token_ids"]],
            )
            for k in chosen
        ]
        for lang in sorted(by_lang)
    }
    return out, excluded


def _energy_mj() -> int | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
    except Exception:
        return None


def _done_cells(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(x)["cell"] for x in path.read_text(encoding="utf-8").splitlines() if x}


def run_block(cfg: dict[str, Any], arm_cfg: dict[str, Any], out_root: Path) -> dict[str, Any]:
    engines, arms, models = load_configs()
    engine, model = cfg["engine"], cfg["model"]
    arm = arm_cfg["arm"]
    label = arm_cfg.get("label", "default")
    block_dir = out_root / f"{engine}-{arm}-{label}-{model}"
    block_dir.mkdir(parents=True, exist_ok=True)
    cells_path = block_dir / "cells.jsonl"
    done = _done_cells(cells_path)

    seed = int(cfg.get("seed", 1234))
    prompts, excluded = load_prompts(
        REPO / "data/bench",
        cfg["split"],
        cfg["languages"],
        cfg["tasks"],
        int(cfg.get("max_model_len", 4096)) - int(cfg["max_tokens"]),
        int(cfg.get("max_prompts_per_cell", 30)),
        seed,
    )
    cells = [
        Cell(lang, c, rep)
        for rep in range(int(cfg["repeats"]))
        for lang in sorted(prompts)
        for c in cfg["concurrency"]
    ]
    # stable across processes (built-in hash() of str is salted per process)
    order_seed = int(hashlib.sha256(f"{seed}|{arm}|{label}".encode()).hexdigest()[:8], 16)
    # block randomisation: shuffle within each repeat, repeats stay in order
    shuffled: list[Cell] = []
    for rep in range(int(cfg["repeats"])):
        rc = [c for c in cells if c.repeat == rep]
        random.Random(order_seed + rep).shuffle(rc)
        shuffled += rc
    todo = [c for c in shuffled if c.key not in done]
    status: dict[str, Any] = {"block": block_dir.name, "todo": len(todo), "done": len(done)}
    if not todo:
        return status

    tgt = models["targets"][model]
    target_path = str(local_snapshot(tgt["repo"], tgt["revision"]))
    drafter = None
    drafter_path = None
    dkey = arms["arms"][arm].get("drafter")
    if dkey:
        dkey = dkey.format(size=model.split("-")[-1])
        drafter = arms["drafters"][dkey]
        drafter_path = str(local_snapshot(drafter["repo"], drafter["revision"]))
    vram = gpu_guard.query()
    frac = cfg.get("gpu_mem_util") or round(
        gpu_guard.max_safe_fraction(vram)
        - float(engines[engine].get("mem_fraction_headroom", 0.01)),
        3,
    )
    gpu_guard.preflight(frac, vram)
    power = gpu_guard.power_state()
    gpu_guard.power_preflight(power, cfg.get("min_enforced_power_w"))
    plan = build_plan(
        engine,
        arm,
        model=model,
        target_path=target_path,
        drafter_path=drafter_path,
        max_model_len=int(cfg.get("max_model_len", 4096)),
        gpu_mem_util=frac,
        seed=seed,
        overrides=arm_cfg.get("spec_override"),
    )
    manifest = build_manifest(
        run_id=f"{cfg['run_name']}/{block_dir.name}",
        config={**cfg, "arm_cfg": arm_cfg, "gpu_mem_util": frac},
        engine=engine,
        engine_version=engines[engine]["version"],
        engine_launch_cmd=plan.cmd,
        target={"repo": tgt["repo"], "revision": tgt["revision"], "dtype": tgt["dtype"]},
        drafter=(
            {"repo": drafter["repo"], "revision": drafter["revision"], "method": arm}
            if drafter
            else None
        ),
        gpu=probe_gpu(),
        seeds={"engine": seed, "order": order_seed, "prompt_selection": seed},
        start_utc=now_utc(),
    )
    manifest["notes"] = json.dumps({"excluded_long_prompts": excluded, "power_state_start": power})
    stamp = time.strftime("%Y%m%dT%H%M%S")
    with (
        TelemetrySampler(block_dir / f"telemetry-{stamp}.csv"),
        Server(plan, block_dir / f"server-{stamp}.log") as srv,
    ):
        # warm-up (not logged): one short request per language
        warm = LoadSpec(
            srv.base_url,
            model,
            max_tokens=16,
            extra_body=dict(engines[engine].get("request_extras") or {}),
        )
        asyncio.run(run_load(warm, [ps[0] for ps in prompts.values() if ps]))
        # Engines with cross-request draft state (SGLang NGRAM keeps one corpus of every earlier
        # request, D-023) are flushed before each cell so a cell never drafts from another
        # cell's outputs. Run config may override the engine default.
        flush_path = engines[engine].get("flush_cache_endpoint")
        flush = bool(cfg.get("flush_cache_between_cells", flush_path is not None)) and flush_path
        for cell in todo:
            if flush:
                import httpx

                httpx.post(f"{srv.base_url}{flush_path}", timeout=60).raise_for_status()
            spec = LoadSpec(
                srv.base_url,
                model,
                max_tokens=int(cfg["max_tokens"]),
                temperature=float(cfg.get("temperature", 0.0)),
                seed=seed,
                ignore_eos=bool(cfg.get("ignore_eos", False)),
                concurrency=cell.concurrency,
                load_mode=cfg.get("load_mode", "closed_loop"),
                request_rate=cfg.get("request_rate"),
                # Engine-level extras, then run-level ones (e.g. top_p / top_k for a sampled run).
                extra_body={
                    **(engines[engine].get("request_extras") or {}),
                    **(cfg.get("request_extras") or {}),
                },
            )
            p0 = gpu_guard.power_state()
            e0, t0 = _energy_mj(), time.time()
            results, wall = asyncio.run(run_load(spec, prompts[cell.language], rng_seed=seed))
            e1, t1 = _energy_mj(), time.time()
            # compress_raw is opt-in (see indicspec.rawio); the default stays plain JSONL.
            compress = bool(cfg.get("compress_raw", False))
            with open_append(block_dir / "requests.jsonl", compress) as fh:
                for r in results:
                    row = to_log_row(
                        r,
                        spec,
                        run_id=manifest["run_id"],
                        arm=arm if label == "default" else f"{arm}.{label}",
                        engine=engine,
                        split=cfg["split"],
                        repeat=cell.repeat,
                        draft_params=arm_cfg.get("spec_override") or {},
                    )
                    validate(row, "request_log.v1")
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            with open(cells_path, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "cell": cell.key,
                            "language": cell.language,
                            "concurrency": cell.concurrency,
                            "repeat": cell.repeat,
                            "start_unix": t0,
                            "end_unix": t1,
                            "wall_s": wall,
                            "energy_mj": (e1 - e0) if e0 is not None and e1 is not None else None,
                            "n_prompts": len(results),
                            "n_errors": sum(1 for r in results if r.error),
                            "total_output_tokens": sum(r.output_tokens for r in results),
                            "enforced_power_limit_w": p0.get("enforced_power_limit_w"),
                            "windows_power_scheme": p0.get("windows_power_scheme"),
                            "cache_flushed": bool(flush),
                        }
                    )
                    + "\n"
                )
    manifest["end_utc"] = now_utc()
    validate(manifest, "run_manifest.v1")
    (block_dir / f"run_manifest-{stamp}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    status["ran"] = len(todo)
    return status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_root = REPO / "results/raw" / cfg["run_name"]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")
    failures = 0
    for arm_cfg in cfg["arms"]:
        try:
            st = run_block(cfg, arm_cfg, out_root)
            print(json.dumps(st))
        except Exception as exc:  # recorded and continued; never silently dropped
            failures += 1
            with open(out_root / "block_failures.jsonl", "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({"arm_cfg": arm_cfg, "error": repr(exc)[:4000], "time": now_utc()})
                    + "\n"
                )
            print(f"block failed: {arm_cfg}: {exc!r}"[:2000])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
