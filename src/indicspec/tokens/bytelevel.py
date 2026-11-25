"""Byte-level BPE vocabulary decoding.

Qwen tokenizers (like GPT-2) are byte-level BPE: every raw byte 0..255 is
mapped to a printable Unicode character before BPE, and vocabulary strings are
written in that alphabet.  This module re-implements the GPT-2
``bytes_to_unicode`` table (Radford et al., openai/gpt-2 ``encoder.py``) and
inverts it so a vocabulary string can be turned back into the exact raw bytes
the token stands for.

The implementation is *verified, not assumed*: ``tests/data/test_qwen_bytelevel.py``
checks on the real Qwen3 tokenizer that concatenating the decoded bytes of
every token in an encoded text reproduces the UTF-8 bytes of that text.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte -> printable-unicode table (256 entries, bijective)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    return {c: b for b, c in bytes_to_unicode().items()}


def token_str_to_bytes(token: str) -> bytes:
    """Decode a byte-level vocabulary string to the raw bytes it encodes.

    Raises KeyError if the string contains a character outside the byte-level
    alphabet (e.g. a special/added token such as ``<|im_start|>``); callers
    should treat those as special tokens.
    """
    table = unicode_to_bytes()
    return bytes(table[ch] for ch in token)


def is_bytelevel_string(token: str) -> bool:
    table = unicode_to_bytes()
    return all(ch in table for ch in token)
