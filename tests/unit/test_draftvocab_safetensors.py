from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from indicspec.tokens.draftvocab import load_draft_vocab, mapping_from_tensors
from indicspec.tokens.safetensors_np import list_tensors, load_tensor


def _write_st(path: Path, tensors: dict[str, tuple[str, np.ndarray, bytes]]) -> None:
    header: dict = {}
    buf = b""
    for name, (dtype, arr, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [len(buf), len(buf) + len(raw)],
        }
        buf += raw
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + buf)


def _bf16_bytes(x: np.ndarray) -> bytes:
    return (x.astype(np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()


def test_reader_roundtrip_including_bf16(tmp_path: Path):
    f32 = np.arange(6, dtype=np.float32).reshape(2, 3)
    bf = np.array([[1.0, -2.5], [0.15625, 3.0]], dtype=np.float32)  # exactly representable in bf16
    i64 = np.array([5, -1, 7], dtype=np.int64)
    p = tmp_path / "m.safetensors"
    _write_st(
        p,
        {
            "a": ("F32", f32, f32.astype("<f4").tobytes()),
            "b": ("BF16", bf, _bf16_bytes(bf)),
            "c": ("I64", i64, i64.astype("<i8").tobytes()),
        },
    )
    assert list_tensors(p)["b"] == ("BF16", [2, 2])
    np.testing.assert_array_equal(load_tensor(p, "a"), f32)
    np.testing.assert_array_equal(load_tensor(p, "b"), bf)
    np.testing.assert_array_equal(load_tensor(p, "c"), i64)
    with pytest.raises(KeyError):
        load_tensor(p, "missing")


def test_mapping_offset_convention():
    t2d = np.zeros(10, dtype=bool)
    t2d[[0, 3, 4, 9]] = True
    target = np.array([0, 3, 4, 9])
    d2t = target - np.arange(4)
    np.testing.assert_array_equal(mapping_from_tensors(d2t, t2d), target)
    with pytest.raises(ValueError):
        mapping_from_tensors(target, t2d)  # absolute ids, not offsets -> rejected
    with pytest.raises(ValueError):
        mapping_from_tensors(d2t[:3], t2d)


def test_load_draft_vocab_from_checkpoint(tmp_path: Path):
    t2d = np.zeros(8, dtype=bool)
    t2d[[1, 2, 6]] = True
    d2t = np.array([1, 2, 6]) - np.arange(3)
    _write_st(
        tmp_path / "model.safetensors",
        {
            "d2t": ("I64", d2t, d2t.astype("<i8").tobytes()),
            "t2d": ("BOOL", t2d, t2d.astype(np.uint8).tobytes()),
        },
    )
    dv = load_draft_vocab(tmp_path)
    assert dv.target_ids == {1, 2, 6} and len(dv) == 3 and dv.target_vocab_size == 8
