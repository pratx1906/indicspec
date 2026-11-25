"""Engine-truth per-position acceptance (coverage-split test, D-010; WS3.5 without a replay model).

vLLM (``--per-request-spec-decode-metrics detailed`` + ``return_token_ids``) gives, per request, the
output token ids and, per verification step, ``per_step_drafted[j]`` and ``per_step_accepted[j]``.

Alignment (verified by :func:`align` against the token count, never assumed):
* output token 0 comes from prefill (no speculation);
* step j starts at position p_j; it emits ``a_j`` accepted draft tokens plus one target token, so
  ``p_{j+1} = p_j + a_j + 1``; the last step may be cut by max_tokens / EOS.

Evaluated positions in step j: ``p_j .. p_j + a_j - 1`` are *accepted*; if ``a_j < d_j`` the draft
was rejected at position ``p_j + a_j`` (the token there is the target's own choice). Positions after
a rejection are not evaluated (chain verification).

For each evaluated position we record the target token, its class, whether it is in the drafter's
vocabulary, and accepted/rejected. Under greedy decoding an out-of-vocab target token can never be
accepted, so ``P(accept | in-vocab)`` is the informative quantity across languages.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Evaluated:
    position: int
    token_id: int
    accepted: bool
    step: int
    depth: int  # 0-based index within the step's draft


class AlignmentError(ValueError):
    pass


def align(
    n_output: int, per_step_drafted: Sequence[int], per_step_accepted: Sequence[int]
) -> list[tuple[int, int, int]]:
    """Return (start_position, drafted, accepted) per step; raise if counts are inconsistent."""
    if len(per_step_drafted) != len(per_step_accepted):
        raise AlignmentError("drafted/accepted arrays differ in length")
    p = 1
    out = []
    for d, a in zip(per_step_drafted, per_step_accepted, strict=True):
        if a < 0 or a > d:
            raise AlignmentError(f"accepted {a} outside [0, {d}]")
        if p >= n_output:
            raise AlignmentError(f"step starts at {p} beyond output length {n_output}")
        out.append((p, d, a))
        p += a + 1
    # The final step may be truncated by max_tokens / EOS, so p may overshoot by at most a+1.
    if p < n_output:
        raise AlignmentError(f"steps cover {p} positions but output has {n_output}")
    return out


def evaluated_positions(
    output_ids: Sequence[int], per_step_drafted: Sequence[int], per_step_accepted: Sequence[int]
) -> list[Evaluated]:
    ev: list[Evaluated] = []
    n = len(output_ids)
    for j, (p, d, a) in enumerate(align(n, per_step_drafted, per_step_accepted)):
        for k in range(a):
            if p + k < n:
                ev.append(Evaluated(p + k, int(output_ids[p + k]), True, j, k))
        if a < d and p + a < n:
            ev.append(Evaluated(p + a, int(output_ids[p + a]), False, j, a))
    return ev


@dataclass
class Tally:
    evaluated: int = 0
    accepted: int = 0

    @property
    def rate(self) -> float:
        return self.accepted / self.evaluated if self.evaluated else float("nan")


def split_by(
    ev: Iterable[Evaluated], key: dict[int, str] | None = None, vocab: set[int] | None = None
) -> dict[str, Tally]:
    """Tally acceptance by token class (``key``) and/or draft-vocab membership (``vocab``)."""
    out: dict[str, Tally] = {}
    for e in ev:
        parts = []
        if key is not None:
            parts.append(key.get(e.token_id, "UNKNOWN"))
        if vocab is not None:
            parts.append("in_vocab" if e.token_id in vocab else "out_vocab")
        k = "|".join(parts) or "ALL"
        t = out.setdefault(k, Tally())
        t.evaluated += 1
        t.accepted += int(e.accepted)
    return out
