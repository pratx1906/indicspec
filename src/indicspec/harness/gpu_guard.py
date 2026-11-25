"""VRAM guard: use the GPU fully, never OOM, never spill to system memory (owner rule, D-014).

On Windows/WSL the NVIDIA driver can fall back to *system* memory when VRAM is over-committed
("CUDA sysmem fallback"), which silently slows kernels and would corrupt speed measurements.
Changing that driver policy is a system-level GPU setting (spec §0 rule 6: ask first), so we
guard from our side instead:

1. **Pre-flight:** the engine's memory budget (fraction x total) must fit in *currently free* VRAM
   minus a margin; otherwise refuse to launch.
2. **In-run:** telemetry peak ``memory.used`` must stay below ``total - margin``. Reaching the
   ceiling is the signature of over-commitment/fallback; such runs are flagged invalid.
No engine is ever started with CPU offload or swap options.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MARGIN_MIB = 512


@dataclass(frozen=True)
class VramState:
    total_mib: int
    used_mib: int

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib


def query(gpu_index: int = 0) -> VramState:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise RuntimeError("nvidia-smi not found")
    out = subprocess.run(
        [
            exe,
            f"--id={gpu_index}",
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    total, used = (int(float(x)) for x in out.split(","))
    return VramState(total, used)


def budget_mib(fraction: float, state: VramState) -> int:
    return int(fraction * state.total_mib)


def preflight(fraction: float, state: VramState, margin_mib: int = DEFAULT_MARGIN_MIB) -> None:
    """Raise if an engine asking for ``fraction`` of total VRAM would not fit in free VRAM."""
    need = budget_mib(fraction, state)
    if need + margin_mib > state.free_mib:
        raise RuntimeError(
            f"VRAM pre-flight failed: engine budget {need} MiB + margin {margin_mib} MiB > "
            f"free {state.free_mib} MiB (used {state.used_mib}/{state.total_mib}). "
            "Lower gpu_mem_util or free VRAM; never rely on sysmem fallback."
        )


def max_safe_fraction(state: VramState, margin_mib: int = DEFAULT_MARGIN_MIB) -> float:
    return max(0.0, (state.free_mib - margin_mib) / state.total_mib)


def in_run_ok(peak_used_mib: float, total_mib: int, margin_mib: int = DEFAULT_MARGIN_MIB) -> bool:
    return peak_used_mib < total_mib - margin_mib


# --------------------------------------------------------------------------- power state (D-018)


def power_state(gpu_index: int = 0) -> dict[str, object]:
    """Enforced power limit (NVML), current clocks and throttle reasons, and the Windows power plan.

    On a laptop the OS power mode can silently cap the GPU (e.g. 55 W in "Silent"), which changes
    every speed and energy number. This is recorded with every run and checked before launching.
    """
    state: dict[str, object] = {}
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        state["enforced_power_limit_w"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000
        state["default_power_limit_w"] = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000
        state["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        state["max_sm_clock_mhz"] = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
        state["temperature_c"] = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        state["throttle_reasons_hex"] = hex(pynvml.nvmlDeviceGetCurrentClocksEventReasons(h))
    except Exception as exc:  # recorded, never guessed
        state["nvml_error"] = repr(exc)
    exe = (
        shutil.which("powercfg.exe")
        or shutil.which("powercfg")
        or next(  # WSL non-login shells may lack the Windows PATH entries
            (p for p in ("/mnt/c/Windows/System32/powercfg.exe",) if Path(p).exists()), None
        )
    )
    if exe:  # Windows, or WSL with interop
        try:
            out = subprocess.run(
                [exe, "/getactivescheme"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            state["windows_power_scheme"] = (
                out.split("(")[-1].rstrip(")").strip() if "(" in out else out
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            state["windows_power_scheme_error"] = repr(exc)
    return state


def power_preflight(state: dict[str, object], min_enforced_w: float | None) -> None:
    """Raise if the enforced GPU power limit is below the configured minimum."""
    if min_enforced_w is None:
        return
    lim = state.get("enforced_power_limit_w")
    if not isinstance(lim, (int, float)):
        raise RuntimeError(f"power pre-flight: enforced power limit unknown ({state})")
    if lim < min_enforced_w:
        raise RuntimeError(
            f"power pre-flight failed: enforced GPU power limit {lim:.0f} W < required "
            f"{min_enforced_w:.0f} W (Windows power plan: {state.get('windows_power_scheme')}). "
            "Switch the laptop to its performance power mode on AC power."
        )
