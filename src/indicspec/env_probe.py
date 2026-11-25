"""Environment probe (WS0).

Writes a JSON description of the machine: GPU (name, VRAM, compute capability,
driver, clocks/power limits), CUDA toolkit, CPU, RAM, disk, OS, Python, and
(if importable) PyTorch.  Every field records the raw command output it came
from so the probe itself is auditable.  Missing tools are recorded as errors,
never guessed.

Usage:  python -m indicspec.env_probe --out env/probe.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

NVSMI_FIELDS = [
    "name",
    "memory.total",
    "driver_version",
    "compute_cap",
    "clocks.max.sm",
    "clocks.max.mem",
    "power.limit",
    "power.max_limit",
    "pcie.link.gen.max",
    "pcie.link.width.max",
    "persistence_mode",
]


def _run(cmd: list[str], timeout: float = 30.0) -> dict[str, Any]:
    """Run a command and capture stdout/stderr/returncode without raising."""
    exe = shutil.which(cmd[0])
    if exe is None:
        return {"cmd": cmd, "ok": False, "error": f"{cmd[0]} not found on PATH"}
    try:
        p = subprocess.run(
            [exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"cmd": cmd, "ok": False, "error": repr(exc)}
    return {
        "cmd": cmd,
        "ok": p.returncode == 0,
        "returncode": p.returncode,
        "stdout": p.stdout.strip(),
        "stderr": p.stderr.strip()[-2000:],
    }


def parse_nvsmi_csv(text: str, fields: list[str]) -> list[dict[str, str]]:
    """Parse `nvidia-smi --query-gpu=... --format=csv,noheader` output."""
    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = [v.strip() for v in line.split(",")]
        if len(values) != len(fields):
            continue
        gpus.append(dict(zip(fields, values, strict=True)))
    return gpus


def probe_gpu() -> dict[str, Any]:
    raw = _run(["nvidia-smi", f"--query-gpu={','.join(NVSMI_FIELDS)}", "--format=csv,noheader"])
    out: dict[str, Any] = {"raw": raw}
    if raw.get("ok"):
        out["gpus"] = parse_nvsmi_csv(raw["stdout"], NVSMI_FIELDS)
    smi = _run(["nvidia-smi"])
    if smi.get("ok"):
        # The banner line contains the driver's max supported CUDA version.
        for line in smi["stdout"].splitlines():
            if "CUDA Version" in line:
                out["driver_cuda_version_line"] = line.strip(" |")
                break
    return out


def probe_nvcc() -> dict[str, Any]:
    raw = _run(["nvcc", "--version"])
    out: dict[str, Any] = {"raw": raw}
    if raw.get("ok"):
        for line in raw["stdout"].splitlines():
            if "release" in line:
                out["release_line"] = line.strip()
    return out


def probe_torch() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # torch is not a dependency of the CPU env
        return {"importable": False, "error": repr(exc)}
    info: dict[str, Any] = {
        "importable": True,
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        info["arch_list"] = torch.cuda.get_arch_list()
    return info


def probe_cpu_mem_disk(repo_root: Path) -> dict[str, Any]:
    import psutil

    cpu_name = platform.processor()
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    elif sys.platform == "win32":
        r = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        if r.get("ok"):
            cpu_name = r["stdout"]
    du = shutil.disk_usage(repo_root)
    vm = psutil.virtual_memory()
    return {
        "cpu_name": cpu_name,
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 2**30, 2),
        "ram_available_gb": round(vm.available / 2**30, 2),
        "disk_path": str(repo_root),
        "disk_total_gb": round(du.total / 2**30, 1),
        "disk_free_gb": round(du.free / 2**30, 1),
    }


def probe_os() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "python": sys.version,
        "python_executable": sys.executable,
    }
    rel = platform.release().lower()
    info["is_wsl"] = "microsoft" in rel or "wsl" in rel
    return info


def build_probe(repo_root: Path) -> dict[str, Any]:
    return {
        "schema": "indicspec.env_probe/1",
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "os": probe_os(),
        "gpu": probe_gpu(),
        "nvcc": probe_nvcc(),
        "torch": probe_torch(),
        "host": probe_cpu_mem_disk(repo_root),
        "env": {k: os.environ[k] for k in ("CUDA_HOME", "HF_HOME") if k in os.environ},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("env/probe.json"))
    args = ap.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    probe = build_probe(repo_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(probe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gpus = probe["gpu"].get("gpus", [])
    print(f"wrote {args.out}; gpus={[g.get('name') for g in gpus]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
