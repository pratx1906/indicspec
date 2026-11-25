"""Vocabulary-level tools: load a tokenizer, classify every entry, coverage.

Tokenizers are fetched with ``huggingface_hub`` at a pinned revision
(only ``tokenizer.json`` / ``tokenizer_config.json``, a few MB) into the
standard HF cache.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from indicspec.tokens.bytelevel import is_bytelevel_string, token_str_to_bytes
from indicspec.tokens.classes import TokenClass, TokenInfo, classify_bytes


@dataclass(frozen=True)
class PinnedTokenizer:
    repo_id: str
    revision: str


def fetch_tokenizer_json(spec: PinnedTokenizer) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(spec.repo_id, "tokenizer.json", revision=spec.revision))


def load_tokenizer(spec: PinnedTokenizer):  # -> tokenizers.Tokenizer
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(fetch_tokenizer_json(spec)))


def vocab_bytes(tokenizer_json: Path) -> tuple[dict[int, bytes], set[int]]:
    """Return ({id: raw bytes} for byte-level entries, {ids of special/added tokens}).

    Reads ``model.vocab`` and ``added_tokens`` straight from tokenizer.json so
    the mapping does not depend on any decoder post-processing.
    """
    data = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
    model = data["model"]
    if model.get("type") != "BPE":
        raise ValueError(f"expected BPE model, got {model.get('type')!r}")
    special = {int(t["id"]) for t in data.get("added_tokens", [])}
    out: dict[int, bytes] = {}
    for tok, idx in model["vocab"].items():
        idx = int(idx)
        if idx in special:
            continue
        if not is_bytelevel_string(tok):
            special.add(idx)
            continue
        out[idx] = token_str_to_bytes(tok)
    return out, special


def classify_vocab(tokenizer_json: Path) -> dict[int, TokenInfo]:
    vb, special = vocab_bytes(tokenizer_json)
    info = {i: classify_bytes(b) for i, b in vb.items()}
    for i in special:
        info[i] = TokenInfo(TokenClass.SPECIAL, 0, 0, False, None)
    return info


def class_counts(info: Mapping[int, TokenInfo]) -> Counter[str]:
    return Counter(str(v.cls) for v in info.values())


def coverage(token_ids: Iterable[int], draft_vocab: set[int]) -> tuple[float, float, int, int]:
    """Frequency-weighted and unique-token coverage of ``token_ids`` by ``draft_vocab``.

    Frequency-weighted: sum_{v in V_draft} f(v) / sum_{v in V} f(v)  (PROJECT_PROMPT App. A).
    Returns (weighted, unique, n_tokens, n_unique).  Empty input -> (nan, nan, 0, 0).
    """
    freq = Counter(token_ids)
    total = sum(freq.values())
    if total == 0:
        return float("nan"), float("nan"), 0, 0
    covered = sum(c for t, c in freq.items() if t in draft_vocab)
    uniq_cov = sum(1 for t in freq if t in draft_vocab)
    return covered / total, uniq_cov / len(freq), total, len(freq)
