"""Bootstrap confidence intervals and Holm correction (docs/PROJECT_BRIEF.md §8.4, App. A)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

Statistic = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class CI:
    estimate: float
    lo: float
    hi: float
    level: float
    n: int
    n_boot: int

    def overlaps(self, other: CI) -> bool:
        return not (self.hi < other.lo or other.hi < self.lo)


def bootstrap_ci(
    values: Sequence[float],
    stat: Statistic = np.mean,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 1234,
) -> CI:
    """Percentile bootstrap: resample with replacement, take the (1-level)/2 quantiles."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.apply_along_axis(stat, 1, x[idx]) if stat is not np.mean else x[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return CI(float(stat(x)), float(lo), float(hi), level, int(x.size), n_boot)


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    stat: Callable[[np.ndarray, np.ndarray], float] = lambda a, b: float(a.mean() / b.mean()),
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 1234,
) -> CI:
    """Paired bootstrap: the same resampled prompt indices are used for both arms.

    Default statistic is the ratio of means (e.g. arm tokens/s over A0 tokens/s).
    """
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    if xa.shape != xb.shape or xa.size == 0:
        raise ValueError("paired samples must be non-empty and the same length")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, xa.size, size=xa.size)
        boots[i] = stat(xa[idx], xb[idx])
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return CI(float(stat(xa, xb)), float(lo), float(hi), level, int(xa.size), n_boot)


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down: returns reject flags in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject
