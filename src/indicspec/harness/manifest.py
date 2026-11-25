"""Run manifest construction (spec §6.2)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_hash(cfg: Any) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def dataset_manifest_sha256() -> str:
    p = REPO / "data" / "MANIFEST.json"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "none"


def now_utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def build_manifest(
    *,
    run_id: str,
    config: dict[str, Any],
    engine: str,
    engine_version: str,
    engine_launch_cmd: list[str] | None,
    target: dict[str, str],
    drafter: dict[str, Any] | None,
    gpu: dict[str, Any],
    seeds: dict[str, Any],
    start_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "run_id": run_id,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "command": sys.argv,
        "config_hash": config_hash(config),
        "config": config,
        "engine": engine,
        "engine_version": engine_version,
        "engine_launch_cmd": engine_launch_cmd,
        "target": target,
        "drafter": drafter,
        "dataset_manifest_sha256": dataset_manifest_sha256(),
        "gpu": gpu,
        "telemetry_csv": None,
        "thermal_summary": None,
        "seeds": seeds,
        "start_utc": start_utc,
        "end_utc": None,
        "notes": None,
    }
