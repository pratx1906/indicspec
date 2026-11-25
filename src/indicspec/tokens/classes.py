"""Token-class classifier (WS3.1).

Each vocabulary entry is classified from its raw bytes *in isolation*.

Classes (proposal from PROJECT_PROMPT.md §WS3.1, refined; see docs/DECISIONS.md D-006):

``SPECIAL``             special / added tokens (not byte-level strings).
``WHITESPACE``          valid UTF-8, every scalar is whitespace.
``ASCII_OR_PUNCT``      valid UTF-8, every non-whitespace scalar is ASCII or a
                        Unicode punctuation/symbol (category P* or S*) – this
                        includes ordinary English words such as " the".
``CHAR``                valid UTF-8, exactly one non-whitespace scalar that is
                        not ASCII/punct (optional surrounding whitespace, as BPE
                        tokens commonly carry a leading space).
``MULTI_CHAR``          valid UTF-8, two or more such scalars.
``BYTE_FRAGMENT_LEAD``  not valid UTF-8; starts on a character boundary but
                        ends with an incomplete multi-byte sequence (the token
                        *opens* a character that a later token completes).
``BYTE_FRAGMENT_CONT``  not valid UTF-8; starts with one or more UTF-8
                        continuation bytes (0x80-0xBF), i.e. it *completes* a
                        character opened by an earlier token.  A token that both
                        starts with continuation bytes and ends incomplete is
                        CONT (its first role in the stream is completion); the
                        ``ends_incomplete`` flag keeps that information.
``BYTE_INVALID``        contains bytes that never occur in well-formed UTF-8
                        (0xC0, 0xC1, 0xF5-0xFF) or other malformed sequences.

Refinement over the prompt's proposal: ``BYTE_INVALID`` is added because the
256 single-byte tokens of a byte-level vocabulary include 13 bytes that can
never appear in UTF-8 text; lumping them into a fragment class would distort
per-class statistics.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class TokenClass(StrEnum):
    SPECIAL = "SPECIAL"
    WHITESPACE = "WHITESPACE"
    ASCII_OR_PUNCT = "ASCII_OR_PUNCT"
    CHAR = "CHAR"
    MULTI_CHAR = "MULTI_CHAR"
    BYTE_FRAGMENT_LEAD = "BYTE_FRAGMENT_LEAD"
    BYTE_FRAGMENT_CONT = "BYTE_FRAGMENT_CONT"
    BYTE_INVALID = "BYTE_INVALID"


FRAGMENT_CLASSES = frozenset(
    {TokenClass.BYTE_FRAGMENT_LEAD, TokenClass.BYTE_FRAGMENT_CONT, TokenClass.BYTE_INVALID}
)

# Unicode blocks for the scripts we study.  Checked against the Unicode
# Character Database bundled with Python (tests/unit/test_token_classes.py::
# test_script_blocks_match_unicodedata); see docs/CLAIMS.md C21.
SCRIPT_BLOCKS: dict[str, tuple[int, int]] = {
    "Devanagari": (0x0900, 0x097F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Latin": (0x0000, 0x024F),
}

LANG_SCRIPT = {
    "hi": "Devanagari",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "en": "Latin",
}


@dataclass(frozen=True)
class TokenInfo:
    cls: TokenClass
    n_bytes: int
    n_scalars: int  # complete scalars fully inside the token
    ends_incomplete: bool
    script: str | None  # dominant script of the (partial) characters, if known


def utf8_seq_len(lead: int) -> int:
    """Expected sequence length for a UTF-8 lead byte; 0 for continuation, -1 invalid."""
    if lead < 0x80:
        return 1
    if 0x80 <= lead <= 0xBF:
        return 0
    if 0xC2 <= lead <= 0xDF:
        return 2
    if 0xE0 <= lead <= 0xEF:
        return 3
    if 0xF0 <= lead <= 0xF4:
        return 4
    return -1  # 0xC0, 0xC1, 0xF5..0xFF


def script_of_codepoint(cp: int) -> str | None:
    for name, (lo, hi) in SCRIPT_BLOCKS.items():
        if lo <= cp <= hi:
            return name
    return None


def script_of_partial(seq: bytes) -> str | None:
    """Script implied by an incomplete 3-byte sequence's first two bytes.

    For U+0800..U+0FFF the UTF-8 form is ``E0 (A0 | cp>>6 & 0x3F) (80 | cp & 0x3F)``,
    so a lead pair ``E0 xx`` pins the 64-code-point sub-block. Two bytes are enough
    to identify every block in SCRIPT_BLOCKS that lies in that range.
    """
    if len(seq) >= 2 and seq[0] == 0xE0 and 0xA0 <= seq[1] <= 0xBF:
        cp_lo = ((seq[1] & 0x3F) << 6) | 0x00
        return script_of_codepoint(cp_lo)
    return None


def _is_ascii_or_punct(ch: str) -> bool:
    if ord(ch) < 0x80:
        return True
    return unicodedata.category(ch)[0] in ("P", "S")


def _split_utf8(b: bytes) -> tuple[int, bytes, int, bool, bool]:
    """Return (n_leading_cont, body, n_trailing_partial, ends_incomplete, invalid)."""
    i = 0
    while i < len(b) and 0x80 <= b[i] <= 0xBF:
        i += 1
    lead_cont = i
    j = i
    invalid = False
    partial = 0
    while j < len(b):
        n = utf8_seq_len(b[j])
        if n <= 0:  # stray continuation or invalid lead in the middle
            invalid = True
            break
        if j + n > len(b):
            # Incomplete tail: remaining bytes must all be continuation bytes.
            tail = b[j + 1 :]
            if all(0x80 <= x <= 0xBF for x in tail):
                partial = len(b) - j
            else:
                invalid = True
            break
        if not all(0x80 <= x <= 0xBF for x in b[j + 1 : j + n]):
            invalid = True
            break
        j += n
    body = b[lead_cont : len(b) - partial] if not invalid else b
    return lead_cont, body, partial, partial > 0, invalid


def classify_bytes(b: bytes) -> TokenInfo:
    """Classify a token from its raw bytes (non-special tokens only)."""
    if len(b) == 0:
        return TokenInfo(TokenClass.SPECIAL, 0, 0, False, None)
    lead_cont, body, partial, ends_incomplete, invalid = _split_utf8(b)
    if invalid or any(utf8_seq_len(x) == -1 for x in b):
        return TokenInfo(TokenClass.BYTE_INVALID, len(b), 0, False, None)
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        # e.g. overlong / surrogate encodings: well-formed shape, invalid value
        return TokenInfo(TokenClass.BYTE_INVALID, len(b), 0, False, None)

    scripts = [s for s in (script_of_codepoint(ord(c)) for c in text if not c.isspace()) if s]
    if partial:
        ps = script_of_partial(b[len(b) - partial :])
        if ps:
            scripts.append(ps)
    script = max(set(scripts), key=scripts.count) if scripts else None

    if lead_cont > 0:
        return TokenInfo(TokenClass.BYTE_FRAGMENT_CONT, len(b), len(text), ends_incomplete, script)
    if ends_incomplete:
        return TokenInfo(TokenClass.BYTE_FRAGMENT_LEAD, len(b), len(text), True, script)

    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return TokenInfo(TokenClass.WHITESPACE, len(b), len(text), False, None)
    if all(_is_ascii_or_punct(c) for c in non_ws):
        return TokenInfo(TokenClass.ASCII_OR_PUNCT, len(b), len(text), False, script)
    cls = TokenClass.CHAR if len(non_ws) == 1 else TokenClass.MULTI_CHAR
    return TokenInfo(cls, len(b), len(text), False, script)
