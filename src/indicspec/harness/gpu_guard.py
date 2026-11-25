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
