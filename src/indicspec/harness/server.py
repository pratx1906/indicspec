"""Launch and stop an inference server from ``configs/engines.yaml`` + ``configs/arms.yaml`` (WS1).

Command construction is a pure function (unit-tested); launching is Linux/WSL only.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

REPO = Path(__file__).resolve().parents[3]


def load_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    def rd(name: str) -> dict[str, Any]:
        return yaml.safe_load((REPO / "configs" / name).read_text(encoding="utf-8"))

    return rd("engines.yaml"), rd("arms.yaml"), rd("models.yaml")


def _fmt(x: Any, values: dict[str, Any]) -> Any:
    if isinstance(x, str):
        return x.format(**values)
    if isinstance(x, dict):
        return {k: _fmt(v, values) for k, v in x.items()}
    if isinstance(x, list):
        return [_fmt(v, values) for v in x]
    return x


@dataclass(frozen=True)
class LaunchPlan:
    engine: str
    arm: str
    cmd: list[str]
    env: dict[str, str]
    port: int
    model: str
    spec_config: Any


def build_plan(
    engine: str,
    arm: str,
    *,
    model: str,
    target_path: str,
    drafter_path: str | None,
    port: int = 8100,
    max_model_len: int = 4096,
    gpu_mem_util: float = 0.85,
    seed: int = 1234,
    max_num_seqs: int = 32,
    overrides: dict[str, Any] | None = None,
    engines: dict[str, Any] | None = None,
    arms: dict[str, Any] | None = None,
) -> LaunchPlan:
    if engines is None or arms is None:
        e_cfg, a_cfg, _ = load_configs()
        engines = engines or e_cfg
        arms = arms or a_cfg
    e = engines[engine]
    a = arms["arms"][arm]
    if engine not in a:
        raise KeyError(f"arm {arm} has no {engine} config")
    venv = os.path.expanduser(e["venv"])
    values = {
        "venv": venv,
        "target_path": target_path,
        "model": model,
        "port": port,
        "max_model_len": max_model_len,
        "gpu_mem_util": gpu_mem_util,
        "seed": seed,
        "max_num_seqs": max_num_seqs,
        "drafter_path": drafter_path or "",
    }
    cmd = [str(c) for c in _fmt(e["serve"], values)]
    spec = _fmt(a[engine], values)
    if overrides:
        if not isinstance(spec, dict):
            raise ValueError("overrides only supported for dict-style (vLLM) speculative configs")
        spec = {**spec, **overrides}
    if engine == "vllm":
        if spec:
            cmd += [str(c) for c in _fmt(e["spec_arg"], {"spec_json": json.dumps(spec)})]
            cmd += list(e.get("per_request_spec_metrics_arg", []))
    elif engine == "sglang":
        cmd += [str(c) for c in (spec or [])]
    else:
        raise ValueError(engine)
    env = {k: str(v) for k, v in (e.get("env") or {}).items()}
    # Engines JIT-compile kernels with tools installed in their venv (e.g. ninja for SGLang),
    # so the venv's bin/ must be first on PATH for the server process.
    env["PATH"] = f"{venv}/bin:" + os.environ.get("PATH", "/usr/bin:/bin")
    env["VIRTUAL_ENV"] = venv
    return LaunchPlan(engine, arm, cmd, env, port, model, spec)


class Server:
    """``with Server(plan, log_path) as s: ...`` — waits for health, kills the process group."""

    def __init__(self, plan: LaunchPlan, log_path: Path, ready_timeout_s: float = 900.0) -> None:
        self.plan = plan
        self.log_path = log_path
        self.ready_timeout_s = ready_timeout_s
        self.proc: subprocess.Popen[bytes] | None = None
        self.base_url = f"http://127.0.0.1:{plan.port}"

    def __enter__(self) -> Server:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            self.plan.cmd,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **self.plan.env},
            start_new_session=True,
        )
        deadline = time.time() + self.ready_timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self._close()
                tail = self.log_path.read_text(errors="replace")[-3000:]
                raise RuntimeError(f"server exited with {self.proc.returncode}:\n{tail}")
            try:
                if httpx.get(self.base_url + "/health", timeout=2).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(2)
        self._close()
        raise TimeoutError(f"server not healthy after {self.ready_timeout_s}s; see {self.log_path}")

    def _close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            if sys.platform == "win32":  # engines are Linux-only; kept for type-checking
                self.proc.kill()
            else:
                os.killpg(self.proc.pid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=30)
        self._log.close()

    def __exit__(self, *exc: object) -> None:
        self._close()
