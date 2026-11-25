"""Minimal safetensors reader on numpy (no torch), including BF16.

Format (https://github.com/huggingface/safetensors): 8-byte little-endian header length N,
N bytes of JSON header ``{name: {dtype, shape, data_offsets: [begin, end]}, "__metadata__": ...}``,
then the byte buffer.  Offsets are relative to the start of the buffer.  BF16 is widened to
float32 by shifting the 16 stored bits into the high half of a uint32 (exact).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

_DTYPES: dict[str, Any] = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
    return header, 8 + n


def list_tensors(path: Path) -> dict[str, tuple[str, list[int]]]:
    header, _ = read_header(path)
    return {k: (v["dtype"], v["shape"]) for k, v in header.items() if k != "__metadata__"}


def load_tensor(path: Path, name: str) -> np.ndarray:
    header, base = read_header(path)
    if name not in header:
        raise KeyError(f"{name!r} not in {path.name}; have {sorted(k for k in header)[:20]}")
    meta = header[name]
    begin, end = meta["data_offsets"]
    shape = meta["shape"]
    dtype = meta["dtype"]
    with open(path, "rb") as fh:
        fh.seek(base + begin)
        raw = fh.read(end - begin)
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        arr = (u16 << 16).view(np.float32)
    elif dtype in _DTYPES:
        arr = np.frombuffer(raw, dtype=np.dtype(_DTYPES[dtype]).newbyteorder("<"))
    else:
        raise ValueError(f"unsupported dtype {dtype}")
    return arr.reshape(shape)


def find_tensor(files: list[Path], name: str) -> tuple[Path, str, list[int]]:
    """Locate a tensor across shards; returns (file, dtype, shape)."""
    for f in files:
        t = list_tensors(f)
        if name in t:
            return f, t[name][0], t[name][1]
    raise KeyError(name)
