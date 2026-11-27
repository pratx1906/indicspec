"""Read raw JSONL logs that may be stored plain, gzipped, or both.

Raw request logs grow to ~50-60 MB per block (GitHub warns above 50 MB). A run can opt in to
writing ``requests.jsonl.gz`` (``compress_raw: true`` in the sweep config), and a finished run can
be packed losslessly with ``scripts/pack_raw.sh``. Every reader goes through ``read_jsonl`` so the
analyses do not care which form is on disk. If both forms exist, the plain rows come first, then
the gzipped rows (appends only ever go to one of them within a run).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import IO, Any


def gz_path(path: Path) -> Path:
    return path.with_name(path.name + ".gz")


def exists(path: Path) -> bool:
    return path.exists() or gz_path(path).exists()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of ``path`` and/or ``path.gz`` (concatenated gzip members are fine); [] if neither."""
    rows: list[dict[str, Any]] = []
    if path.exists():
        rows += [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    gz = gz_path(path)
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as fh:
            rows += [json.loads(x) for x in fh if x.strip()]
    return rows


def open_append(path: Path, compress: bool) -> IO[str]:
    """Text handle appending to ``path`` or, if ``compress``, to ``path.gz`` (a new gzip member)."""
    if compress:
        return gzip.open(gz_path(path), "at", encoding="utf-8")
    return open(path, "a", encoding="utf-8")
