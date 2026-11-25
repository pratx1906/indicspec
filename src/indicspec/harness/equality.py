"""Greedy-equality comparison (spec §9, correctness layer).

At batch size 1 with deterministic settings, an arm's output token ids must equal A0's. On a
mismatch we report the first divergence position; the caller then teacher-forces the target on the
shared prefix to get the top-2 logit gap at that position and classify the mismatch as a near-tie
numerical effect or a bug (threshold pre-registered in docs/PREREG.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Divergence:
    prompt_id: str
    position: int  # index of first differing output token; == min length if one is a prefix
    baseline_token: int | None
    arm_token: int | None
    baseline_len: int
    arm_len: int


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    """Index of the first differing element, or None if equal (same length, same ids)."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def compare(prompt_id: str, baseline: Sequence[int], arm: Sequence[int]) -> Divergence | None:
    pos = first_divergence(baseline, arm)
    if pos is None:
        return None
    return Divergence(
        prompt_id,
        pos,
        baseline[pos] if pos < len(baseline) else None,
        arm[pos] if pos < len(arm) else None,
        len(baseline),
        len(arm),
    )


def classify(top2_logit_gap: float, near_tie_threshold: float) -> str:
    """'near_tie' if the target's top-2 logit gap at the divergence is below the threshold."""
    return "near_tie" if top2_logit_gap < near_tie_threshold else "suspected_bug"
