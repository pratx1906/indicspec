"""Metric definitions (docs/PROJECT_BRIEF.md §8.3 and Appendix A).

The code must match the spec's definitions exactly; any deviation is
documented in docs/DECISIONS.md.  All functions are pure.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def acceptance_length(accepted_draft_tokens: int, verify_steps: int) -> float:
    """tau = (A + S) / S: mean tokens emitted per target verification step.

    A = total accepted draft tokens, S = number of verification steps (each
    emits one bonus token).  No speculation (A = 0) gives tau = 1.
    """
    if verify_steps <= 0:
        raise ValueError("verify_steps must be positive")
    if accepted_draft_tokens < 0:
        raise ValueError("accepted_draft_tokens must be non-negative")
    return (accepted_draft_tokens + verify_steps) / verify_steps


def acceptance_rate(accepted_draft_tokens: int, proposed_draft_tokens: int) -> float:
    """Fraction of proposed draft tokens that were accepted."""
    if proposed_draft_tokens <= 0:
        return float("nan")
    return accepted_draft_tokens / proposed_draft_tokens


def tpot(e2e_latency_s: float, ttft_s: float, output_tokens: int) -> float:
    """Time per output token after the first: (e2e - TTFT) / (n_out - 1)."""
    if output_tokens < 2:
        return float("nan")
    return (e2e_latency_s - ttft_s) / (output_tokens - 1)


def per_user_speed(tpot_s: float) -> float:
    """1 / TPOT, tokens per second for one user."""
    if not tpot_s or math.isnan(tpot_s) or tpot_s <= 0:
        return float("nan")
    return 1.0 / tpot_s


def request_tokens_per_s(output_tokens: int, e2e_latency_s: float) -> float:
    if e2e_latency_s <= 0:
        return float("nan")
    return output_tokens / e2e_latency_s


def speedup(arm_tokens_per_s: float, baseline_tokens_per_s: float) -> float:
    """Per-request tokens/s of the arm divided by A0's on the same prompt/cell."""
    if baseline_tokens_per_s <= 0:
        return float("nan")
    return arm_tokens_per_s / baseline_tokens_per_s


def system_throughput(total_output_tokens: int, wall_time_s: float) -> float:
    if wall_time_s <= 0:
        return float("nan")
    return total_output_tokens / wall_time_s


def time_per_100_words(latency_s: float, output_words: int) -> float:
    """latency_s / (output_words / 100); whitespace words (crude for agglutinative langs)."""
    if output_words <= 0:
        return float("nan")
    return latency_s / (output_words / 100.0)


def whitespace_words(text: str) -> int:
    return len(text.split())


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' method), q in [0, 100]."""
    xs = sorted(v for v in values if not math.isnan(v))
    if not xs:
        return float("nan")
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def predicted_speedup(tau: float, t_draft: float, t_verify: float, t_overhead: float) -> float:
    """Cost model: speedup ~= tau / (1 + (T_draft + T_overhead) / T_verify).

    Expressed relative to one target decode step with T_verify ~= T_AR
    (Appendix A).  When T_verify differs from T_AR use ``predicted_speedup_ar``.
    """
    if t_verify <= 0:
        raise ValueError("t_verify must be positive")
    return tau / (1.0 + (t_draft + t_overhead) / t_verify)


def predicted_speedup_ar(
    tau: float, t_draft: float, t_verify: float, t_overhead: float, t_ar: float
) -> float:
    """General form: tau * T_AR / (T_draft + T_verify + T_overhead)."""
    denom = t_draft + t_verify + t_overhead
    if denom <= 0:
        raise ValueError("iteration time must be positive")
    return tau * t_ar / denom
