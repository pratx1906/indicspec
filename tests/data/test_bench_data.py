"""Checks on the built IndicSpec-Bench files (WS2 acceptance).

Marked ``data``: needs ``data/bench/*.jsonl`` (``make data`` or
``uv run python -m indicspec.data.build --verify``) and the pinned Qwen3 tokenizer.
"""

from __future__ import annotations

import json
import statistics
import unicodedata
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import yaml

from indicspec.data.build import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    lang_script_fraction,
    sha256_file,
)

pytestmark = pytest.mark.data

MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.json"
DATA_DIR = REPO_ROOT / "data"
CFG = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
FILES = sorted(MANIFEST["files"])
TASKS = sorted(CFG["tasks"])

# Script sanity (D-013). Fractions count only letters/marks (L*, M*): spaces, punctuation
# (incl. danda), digits and symbols are ignored. Observed lows are untranslated Latin-script
# named entities (e.g. "George W. Bush", wine grape names), which are legitimate, so a single
# per-row 0.8 cut would reject correct rows. We use three layers instead:
ROW_FLOOR = 0.3  # any row below this is in the wrong script / misaligned
CELL_ROW_THRESHOLD = 0.8  # ... and at least CELL_MIN_SHARE of a (task, lang) cell must reach it
CELL_MIN_SHARE = 0.98
CELL_MEDIAN_MIN = 0.9


def _path(name: str) -> Path:
    return DATA_DIR / name


@cache
def rows_of(name: str) -> tuple[dict[str, Any], ...]:
    p = _path(name)
    if not p.exists():
        pytest.skip(f"{p} missing: run `uv run python -m indicspec.data.build --verify`")
    with open(p, encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f)


def test_manifest_lists_every_task() -> None:
    assert sorted(Path(n).stem for n in FILES) == TASKS


@pytest.mark.parametrize("name", FILES)
def test_checksum_matches_manifest(name: str) -> None:
    p = _path(name)
    if not p.exists():
        pytest.skip(f"{p} missing")
    assert sha256_file(p) == MANIFEST["files"][name]["sha256"]
    assert len(rows_of(name)) == MANIFEST["files"][name]["rows"]


@pytest.mark.parametrize("name", FILES)
def test_parallel_alignment_and_no_duplicates(name: str) -> None:
    rows = rows_of(name)
    task = rows[0]["task"]
    langs = set(CFG["tasks"][task]["langs"])
    by_item: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        assert r["task"] == task
        by_item[r["item_id"]].append(r["lang"])
    for item, item_langs in by_item.items():
        assert len(item_langs) == len(set(item_langs)), f"duplicate (item, lang) in {item}"
        assert set(item_langs) == langs, f"{item}: {sorted(item_langs)}"


@pytest.mark.parametrize("name", FILES)
def test_each_item_in_exactly_one_split(name: str) -> None:
    splits: dict[str, set[str]] = defaultdict(set)
    for r in rows_of(name):
        splits[r["item_id"]].add(r["split"])
        assert r["split"] in {"pilot", "tuning", "report"}
    assert all(len(s) == 1 for s in splits.values())
    report = sum(1 for s in splits.values() if s == {"report"})
    assert report <= CFG["splits"]["report_cap"]


def test_same_item_same_split_across_files() -> None:
    seen: dict[str, str] = {}
    for name in FILES:
        for r in rows_of(name):
            assert seen.setdefault(r["item_id"], r["split"]) == r["split"], r["item_id"]


@pytest.mark.parametrize("name", FILES)
def test_required_fields_and_counts(name: str) -> None:
    sources = {s["repo"]: s for s in CFG["sources"].values()}
    for r in rows_of(name):
        assert r["licence"], "licence field must be set"
        src = sources[r["source_repo"]]
        assert r["licence"] == src["licence"]
        assert r["source_revision"] == src["revision"]
        assert isinstance(r["translationese"], bool)
        assert r["translationese"] == (r["task"] == "chat" and r["lang"] != "en")
        assert [m["role"] for m in r["messages"]] == ["user"]
        # Qwen3 tokenizer normalizer is NFC; the builder stores NFC text (D-013)
        assert unicodedata.normalize("NFC", r["reference_text"]) == r["reference_text"]
        assert r["n_chars"] == len(r["reference_text"]) > 0
        assert r["n_words"] == len(r["reference_text"].split())
        assert r["n_prompt_tokens"] == len(r["prompt_token_ids"]) > 0
        assert r["n_ref_tokens"] == len(r["ref_token_ids"]) > 0


@pytest.mark.parametrize("name", FILES)
def test_script_block_sanity(name: str) -> None:
    by_lang: dict[str, list[float]] = defaultdict(list)
    for r in rows_of(name):
        if r["lang"] == "en":
            continue
        frac = lang_script_fraction(r["reference_text"], r["lang"])
        assert frac >= ROW_FLOOR, (r["item_id"], r["lang"], frac)
        by_lang[r["lang"]].append(frac)
    for lang, fr in by_lang.items():
        share = sum(f >= CELL_ROW_THRESHOLD for f in fr) / len(fr)
        assert share >= CELL_MIN_SHARE, (lang, share)
        assert statistics.median(fr) >= CELL_MEDIAN_MIN, lang


def test_summarisation_disjoint_from_translation() -> None:
    passages = [
        r["reference_text"] for r in rows_of("bench/summarisation.jsonl") if r["lang"] == "en"
    ]
    for r in rows_of("bench/translation.jsonl"):
        if r["lang"] != "en":
            continue
        assert not any(r["reference_text"] in p for p in passages), r["item_id"]


@pytest.fixture(scope="module")
def chat_tok() -> Any:
    from indicspec.data.build import load_chat_tokenizer

    return load_chat_tokenizer(CFG)


@pytest.mark.slow
@pytest.mark.parametrize("name", FILES)
def test_prompt_tokens_roundtrip(name: str, chat_tok: Any) -> None:
    for r in rows_of(name):
        text = chat_tok.chat_text(r["messages"])
        assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert chat_tok.decode(r["prompt_token_ids"]) == text
        assert chat_tok.encode(text) == r["prompt_token_ids"]
        assert chat_tok.decode(r["ref_token_ids"]) == r["reference_text"]
