"""Unit + property tests for byte-level decoding and the token-class classifier."""

from __future__ import annotations

import unicodedata

import pytest
from hypothesis import given
from hypothesis import strategies as st

from indicspec.tokens.bytelevel import bytes_to_unicode, token_str_to_bytes, unicode_to_bytes
from indicspec.tokens.classes import (
    SCRIPT_BLOCKS,
    TokenClass,
    classify_bytes,
    script_of_partial,
    utf8_seq_len,
)

# ---------------------------------------------------------------- byte-level map


def test_bytes_to_unicode_is_bijective_over_256_bytes():
    m = bytes_to_unicode()
    assert sorted(m) == list(range(256))
    assert len(set(m.values())) == 256
    assert unicode_to_bytes() == {v: k for k, v in m.items()}


def test_known_gpt2_mappings():
    m = bytes_to_unicode()
    assert m[ord(" ")] == "Ġ"  # U+0120
    assert m[ord("\n")] == "Ċ"  # U+010A
    assert m[ord("A")] == "A"


@given(st.binary(max_size=64))
def test_roundtrip_bytes_through_alphabet(b: bytes):
    m = bytes_to_unicode()
    s = "".join(m[x] for x in b)
    assert token_str_to_bytes(s) == b


# ---------------------------------------------------------------- classifier: hand cases

# One representative letter per script, and its UTF-8 bytes.
LETTERS = {
    "Tamil": "க",  # U+0B95
    "Telugu": "క",  # U+0C15
    "Kannada": "ಕ",  # U+0C95
    "Malayalam": "ക",  # U+0D15
    "Devanagari": "क",  # U+0915
}


@pytest.mark.parametrize("script,ch", list(LETTERS.items()))
def test_whole_char(script: str, ch: str):
    lo, hi = SCRIPT_BLOCKS[script]
    assert lo <= ord(ch) <= hi
    b = ch.encode()
    assert len(b) == 3
    info = classify_bytes(b)
    assert info.cls is TokenClass.CHAR
    assert info.script == script
    # leading space is still CHAR
    assert classify_bytes(b" " + b).cls is TokenClass.CHAR


@pytest.mark.parametrize("script,ch", list(LETTERS.items()))
def test_fragments(script: str, ch: str):
    b = ch.encode()
    lead1, lead2 = classify_bytes(b[:1]), classify_bytes(b[:2])
    assert lead1.cls is TokenClass.BYTE_FRAGMENT_LEAD
    assert lead2.cls is TokenClass.BYTE_FRAGMENT_LEAD
    assert lead2.script == script  # two lead bytes pin the block
    assert lead1.script is None  # E0 alone is ambiguous
    assert classify_bytes(b[2:]).cls is TokenClass.BYTE_FRAGMENT_CONT
    assert classify_bytes(b[1:]).cls is TokenClass.BYTE_FRAGMENT_CONT
    # completes one char and opens the next
    mixed = classify_bytes(b[2:] + b[:2])
    assert mixed.cls is TokenClass.BYTE_FRAGMENT_CONT and mixed.ends_incomplete


@pytest.mark.parametrize("script,ch", list(LETTERS.items()))
def test_multi_char(script: str, ch: str):
    assert classify_bytes((ch * 2).encode()).cls is TokenClass.MULTI_CHAR
    # complete char followed by a partial one is a LEAD fragment
    assert classify_bytes(ch.encode() + ch.encode()[:1]).cls is TokenClass.BYTE_FRAGMENT_LEAD


def test_combining_vowel_sign_is_its_own_scalar():
    # Tamil "கி" = KA U+0B95 + VOWEL SIGN I U+0BBF -> two scalars
    assert classify_bytes("கி".encode()).cls is TokenClass.MULTI_CHAR
    assert classify_bytes("ி".encode()).cls is TokenClass.CHAR


@pytest.mark.parametrize(
    "text,cls",
    [
        (" the", TokenClass.ASCII_OR_PUNCT),
        ("Hello", TokenClass.ASCII_OR_PUNCT),
        (",", TokenClass.ASCII_OR_PUNCT),
        ("।", TokenClass.ASCII_OR_PUNCT),  # Devanagari danda is punctuation (Po)
        ("  ", TokenClass.WHITESPACE),
        ("\n\n", TokenClass.WHITESPACE),
        ("é", TokenClass.CHAR),
        ("中文", TokenClass.MULTI_CHAR),
    ],
)
def test_misc_classes(text: str, cls: TokenClass):
    assert classify_bytes(text.encode()).cls is cls


@pytest.mark.parametrize("b", [b"\xc0", b"\xc1", b"\xf5", b"\xff", b"\xe0\x41"])
def test_invalid(b: bytes):
    assert classify_bytes(b).cls is TokenClass.BYTE_INVALID


def test_danda_category_assumption():
    assert unicodedata.category("।") == "Po"


@pytest.mark.parametrize("script", ["Devanagari", "Tamil", "Telugu", "Kannada", "Malayalam"])
def test_script_blocks_match_unicodedata(script: str):
    """Every assigned code point in the block is named after the script; neighbours are not."""
    lo, hi = SCRIPT_BLOCKS[script]
    names = [unicodedata.name(chr(cp), "") for cp in range(lo, hi + 1)]
    assigned = [n for n in names if n]
    assert len(assigned) > 60
    assert all(n.startswith(script.upper() + " ") for n in assigned)
    for cp in (lo - 1, hi + 1):
        assert not unicodedata.name(chr(cp), "").startswith(script.upper() + " ")


def test_utf8_seq_len_table():
    assert utf8_seq_len(0x41) == 1
    assert utf8_seq_len(0x80) == 0
    assert utf8_seq_len(0xC2) == 2
    assert utf8_seq_len(0xE0) == 3
    assert utf8_seq_len(0xF0) == 4
    assert utf8_seq_len(0xF8) == -1


def test_script_of_partial_blocks():
    # Every code point's first two UTF-8 bytes map back to its block.
    for script, (lo, hi) in SCRIPT_BLOCKS.items():
        if script == "Latin":
            continue
        for cp in range(lo, hi + 1):
            assert script_of_partial(chr(cp).encode()[:2]) == script


# ---------------------------------------------------------------- properties


@given(st.text(max_size=20))
def test_valid_utf8_never_fragment(s: str):
    b = s.encode("utf-8", errors="surrogatepass")
    if not b:
        return
    try:
        b.decode("utf-8")
    except UnicodeDecodeError:  # lone surrogates
        return
    assert classify_bytes(b).cls not in {
        TokenClass.BYTE_FRAGMENT_LEAD,
        TokenClass.BYTE_FRAGMENT_CONT,
        TokenClass.BYTE_INVALID,
    }


@given(st.binary(min_size=1, max_size=12))
def test_classifier_total_and_consistent(b: bytes):
    info = classify_bytes(b)
    assert isinstance(info.cls, TokenClass)
    assert info.cls is not TokenClass.SPECIAL
    try:
        b.decode("utf-8")
        valid = True
    except UnicodeDecodeError:
        valid = False
    if valid:
        assert info.cls in {
            TokenClass.WHITESPACE,
            TokenClass.ASCII_OR_PUNCT,
            TokenClass.CHAR,
            TokenClass.MULTI_CHAR,
        }
    else:
        assert info.cls in {
            TokenClass.BYTE_FRAGMENT_LEAD,
            TokenClass.BYTE_FRAGMENT_CONT,
            TokenClass.BYTE_INVALID,
        }


@given(st.text(alphabet=st.sampled_from(list("கதசமனவிுா ")), min_size=1, max_size=10))
def test_splitting_valid_text_yields_fragments_at_non_boundaries(s: str):
    b = s.encode()
    for cut in range(1, len(b)):
        left, right = b[:cut], b[cut:]
        on_boundary = (right[0] & 0xC0) != 0x80
        if not on_boundary:
            assert classify_bytes(left).cls is TokenClass.BYTE_FRAGMENT_LEAD
            assert classify_bytes(right).cls is TokenClass.BYTE_FRAGMENT_CONT
