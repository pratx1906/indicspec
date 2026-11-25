"""Draft-vocabulary extraction for pruned-vocabulary EAGLE-3 heads (WS3.3).

EAGLE-3 checkpoints (speculators / SpecForge style) carry two mapping tensors:

* ``t2d``: bool[target_vocab]; True where the target token is in the draft vocabulary.
* ``d2t``: int[draft_vocab]; *offset* such that ``target_id = draft_id + d2t[draft_id]``.

This structure is an assumption to verify, not to trust: :func:`load_draft_vocab` checks that
``t2d.sum() == len(d2t)`` and that the ids recovered from ``d2t`` are exactly ``nonzero(t2d)`` (in
increasing order). If either check fails it raises rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from indicspec.tokens.safetensors_np import list_tensors, load_tensor


@dataclass(frozen=True)
class DraftVocab:
    draft_to_target: np.ndarray  # int64[draft_vocab]: target id for each draft id
    target_vocab_size: int
    source: str

    @property
    def target_ids(self) -> set[int]:
        return set(int(x) for x in self.draft_to_target)

    def __len__(self) -> int:
        return int(self.draft_to_target.size)


def _find(files: list[Path], name: str) -> Path | None:
    for f in files:
        if name in list_tensors(f):
            return f
    return None


def mapping_from_tensors(d2t: np.ndarray, t2d: np.ndarray) -> np.ndarray:
    """Return draft->target ids after verifying the offset convention against ``t2d``."""
    d2t = np.asarray(d2t).astype(np.int64).ravel()
    t2d = np.asarray(t2d).astype(bool).ravel()
    if int(t2d.sum()) != d2t.size:
        raise ValueError(f"t2d has {int(t2d.sum())} True entries but d2t has {d2t.size}")
    ids = np.arange(d2t.size, dtype=np.int64) + d2t
    expected = np.nonzero(t2d)[0].astype(np.int64)
    if not np.array_equal(ids, expected):
        raise ValueError("d2t offsets do not reproduce nonzero(t2d); mapping convention differs")
    return ids


def load_draft_vocab(checkpoint_dir: Path) -> DraftVocab:
    files = sorted(Path(checkpoint_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {checkpoint_dir}")
    fd, ft = _find(files, "d2t"), _find(files, "t2d")
    if fd is None or ft is None:
        raise KeyError("checkpoint lacks d2t/t2d tensors (full-vocabulary drafter?)")
    t2d = load_tensor(ft, "t2d")
    ids = mapping_from_tensors(load_tensor(fd, "d2t"), t2d)
    return DraftVocab(ids, int(t2d.size), str(checkpoint_dir))
