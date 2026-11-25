"""Verify the byte-level decoder and classifier against the real Qwen3 tokenizer.

Marked ``data``: downloads tokenizer.json (a few MB) at the pinned revision.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from indicspec.tokens.classes import TokenClass
from indicspec.tokens.vocab import (
    PinnedTokenizer,
    classify_vocab,
    fetch_tokenizer_json,
    load_tokenizer,
    vocab_bytes,
)

pytestmark = pytest.mark.data

CFG = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/models.yaml").read_text())
SPEC = PinnedTokenizer(CFG["targets"]["qwen3-8b"]["repo"], CFG["targets"]["qwen3-8b"]["revision"])

SAMPLES = {
    "en": "The quick brown fox jumps over the lazy dog. It's 2026!",
    "hi": "भारत एक विशाल देश है जिसकी संस्कृति बहुत पुरानी है।",
    "ta": "தமிழ் உலகின் மிகப் பழமையான மொழிகளில் ஒன்றாகும்.",
    "te": "తెలుగు భారతదేశంలో ఎక్కువగా మాట్లాడే భాషలలో ఒకటి.",
    "kn": "ಕನ್ನಡ ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಅಧಿಕೃತ ಭಾಷೆಯಾಗಿದೆ.",
    "ml": "മലയാളം കേരളത്തിലെ ഔദ്യോഗിക ഭാഷയാണ്.",
}


@pytest.fixture(scope="module")
def tok_json() -> Path:
    return fetch_tokenizer_json(SPEC)


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(SPEC)


def test_model_is_bytelevel_bpe(tok_json: Path):
    import json

    data = json.loads(tok_json.read_text(encoding="utf-8"))
    assert data["model"]["type"] == "BPE"
    kinds = [data["pre_tokenizer"]["type"]] + [
        p["type"] for p in data["pre_tokenizer"].get("pretokenizers", [])
    ]
    assert "ByteLevel" in kinds
    assert data["decoder"]["type"] == "ByteLevel"


@pytest.mark.parametrize("lang", list(SAMPLES))
def test_concatenated_token_bytes_equal_text(tok, tok_json: Path, lang: str):
    vb, _special = vocab_bytes(tok_json)
    text = SAMPLES[lang]
    ids = tok.encode(text, add_special_tokens=False).ids
    assert b"".join(vb[i] for i in ids) == text.encode("utf-8")


def test_classes_partition_vocab(tok_json: Path):
    vb, special = vocab_bytes(tok_json)
    info = classify_vocab(tok_json)
    assert set(info) == set(vb) | special
    assert not (set(vb) & special)
    assert all(isinstance(v.cls, TokenClass) for v in info.values())
    # every single byte 0..255 is a token in a byte-level vocab
    singles = {b for b in vb.values() if len(b) == 1}
    assert len(singles) == 256
