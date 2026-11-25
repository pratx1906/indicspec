"""Build the frozen IndicSpec-Bench (WS2) from pinned Hugging Face dataset revisions.

Usage (from the repo root)::

    uv run python -m indicspec.data.build            # build data/bench/*.jsonl + data/MANIFEST.json
    uv run python -m indicspec.data.build --verify   # rebuild data/bench and check MANIFEST sha256

Sources, revisions, licences, prompt templates and split fractions live in
``configs/data_sources.yaml``. Raw downloads go to ``data/raw/<org>__<name>/`` (gitignored).
``data/bench/*.jsonl`` is gitignored too: the repository ships this builder and the checksummed
MANIFEST, not the source text (docs/DECISIONS.md D-009, D-013).

Every row is one (item, language) pair; ``item_id`` is shared across the languages of a
parallel task and the split is assigned per item, so all languages of an item share it.
Output is deterministic: rows sorted by (item_id, lang), JSON with ``sort_keys`` and
``ensure_ascii=False``, LF line endings.

Pure helpers (split assignment, passage alignment, script fractions, JSON lines) import no
heavy dependencies; transformers / pyarrow / huggingface_hub are imported lazily so the CPU
unit tests run without the ``data`` extra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from indicspec.tokens.classes import LANG_SCRIPT, SCRIPT_BLOCKS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data_sources.yaml"
MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"
DEFAULT_RAW = REPO_ROOT / "data" / "raw"
DEFAULT_BENCH = REPO_ROOT / "data" / "bench"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "MANIFEST.json"
MANIFEST_SCHEMA_VERSION = 1
SPLITS = ("pilot", "tuning", "report")

# --------------------------------------------------------------------------- pure helpers


def normalize_ws(text: str) -> str:
    """Collapse every whitespace run to one space and strip the ends."""
    return " ".join(text.split())


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(text: str, n: int = 12) -> str:
    return sha256_bytes(text.encode("utf-8"))[:n]


def seeded_rank(keys: Iterable[str], seed: int | str, salt: str = "") -> list[str]:
    """Order ``keys`` by sha256("<seed>:<salt>:<key>") (ties broken by key). Deterministic."""
    uniq = set(keys)
    return sorted(uniq, key=lambda k: (sha256_bytes(f"{seed}:{salt}:{k}".encode()), k))


@dataclass(frozen=True)
class SplitResult:
    assignment: dict[str, str]  # item_id -> split
    dropped_by_cap: tuple[str, ...]


def assign_splits(
    item_ids: Iterable[str],
    seed: int | str,
    pilot_frac: float,
    tuning_frac: float,
    report_cap: int | None,
) -> SplitResult:
    """Item-level split assignment: pilot / tuning / report (capped).

    Items are ranked by a seeded hash, the first ``floor(pilot_frac*n + 0.5)`` are pilot, the
    next ``floor(tuning_frac*n + 0.5)`` tuning, then up to ``report_cap`` report; the rest are
    dropped. The result depends only on the set of ids and the seed, not on input order.
    """
    if not (pilot_frac >= 0 and tuning_frac >= 0 and pilot_frac + tuning_frac <= 1):
        raise ValueError("split fractions must be non-negative and sum to <= 1")
    ranked = seeded_rank(item_ids, seed, "split")
    n = len(ranked)
    n_pilot = int(pilot_frac * n + 0.5)
    n_tuning = int(tuning_frac * n + 0.5)
    assignment: dict[str, str] = {}
    dropped: list[str] = []
    for k, item in enumerate(ranked):
        if k < n_pilot:
            assignment[item] = "pilot"
        elif k < n_pilot + n_tuning:
            assignment[item] = "tuning"
        elif report_cap is None or k - n_pilot - n_tuning < report_cap:
            assignment[item] = "report"
        else:
            dropped.append(item)
    return SplitResult(assignment, tuple(sorted(dropped)))


def locate_sentences(passage: str, sentences: Iterable[str]) -> list[str]:
    """Sentences that occur verbatim in ``passage``, in passage order, without overlaps.

    On overlap the earlier (then longer) match wins. Inputs should be whitespace-normalised.
    """
    hits = sorted((passage.find(s), -len(s), s) for s in set(sentences) if s and s in passage)
    out: list[str] = []
    end = -1
    for pos, _neg_len, s in hits:
        if pos >= end:
            out.append(s)
            end = pos + len(s)
    return out


def script_fraction(text: str, script: str) -> tuple[float, int]:
    """Fraction of letters/marks in ``text`` that fall in ``script``'s Unicode block.

    Only categories L* and M* are counted (spaces, punctuation incl. danda, digits and symbols
    are ignored). Returns (fraction, number of counted characters); fraction is nan if none.
    """
    lo, hi = SCRIPT_BLOCKS[script]
    n = k = 0
    for ch in text:
        if unicodedata.category(ch)[0] not in ("L", "M"):
            continue
        n += 1
        if lo <= ord(ch) <= hi:
            k += 1
    return (k / n if n else float("nan")), n


def lang_script_fraction(text: str, lang: str) -> float:
    return script_fraction(text, LANG_SCRIPT[lang])[0]


def text_stats(text: str) -> dict[str, int]:
    return {
        "n_chars": len(text),
        "n_words": len(text.split()),
        "n_bytes": len(text.encode("utf-8")),
    }


def json_line(row: Mapping[str, Any]) -> str:
    """Stable single-line JSON serialisation used for every bench row."""
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def readme_licence(readme_text: str) -> str | None:
    """``license`` from a Hugging Face README YAML front matter (lower-cased), or None."""
    text = readme_text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    if end < 0:
        return None
    meta = yaml.safe_load(text[4:end]) or {}
    lic = meta.get("license")
    if isinstance(lic, list):
        lic = ",".join(str(x) for x in lic)
    return str(lic).lower() if lic is not None else None


def render(template: str, **kw: str) -> str:
    return template.format(**kw)


# --------------------------------------------------------------------------- data structures


@dataclass
class Cell:
    """One (item, language) before tokenisation."""

    prompt: str
    reference_text: str
    source_row_ref: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)
    translationese: bool = False


@dataclass
class TaskBuild:
    task: str
    source_key: str
    items: dict[str, dict[str, Cell]]  # item_id -> lang -> cell
    stats: dict[str, Any]


# --------------------------------------------------------------------------- download


class Fetcher:
    """Downloads files of a dataset repo at a pinned revision into ``raw_root``."""

    def __init__(self, raw_root: Path) -> None:
        self.raw_root = raw_root
        self.fetched: dict[str, dict[str, str]] = defaultdict(dict)  # repo -> file -> sha256

    def __call__(self, repo: str, revision: str, filename: str) -> Path:
        from huggingface_hub import hf_hub_download

        local_dir = self.raw_root / repo.replace("/", "__")
        path = Path(
            hf_hub_download(
                repo, filename, repo_type="dataset", revision=revision, local_dir=local_dir
            )
        )
        self.fetched[repo][filename] = sha256_file(path)
        return path


def check_licence(fetch: Fetcher, src: Mapping[str, Any]) -> str:
    readme = fetch(src["repo"], src["revision"], "README.md").read_text(encoding="utf-8")
    found = readme_licence(readme)
    expected = str(src["licence"]).lower()
    if found != expected:
        raise RuntimeError(
            f"{src['repo']}@{src['revision']}: README licence {found!r} != config {expected!r}"
        )
    return found


def _load_examples(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "examples" not in data:
        raise ValueError(f"{path}: expected an object with an 'examples' list")
    return list(data["examples"])


# --------------------------------------------------------------------------- FLORES / Belebele


def _belebele_passages(
    fetch: Fetcher, src: Mapping[str, Any], langs: Sequence[str]
) -> tuple[dict[str, dict[str, tuple[str, int]]], dict[str, str]]:
    """lang -> link -> (passage, first row index); plus lang -> file name."""
    out: dict[str, dict[str, tuple[str, int]]] = {}
    files: dict[str, str] = {}
    for lang in langs:
        code = src["lang_codes"][lang]
        fname = f"data/{code}.jsonl"
        files[lang] = fname
        per: dict[str, tuple[str, int]] = {}
        lines = fetch(src["repo"], src["revision"], fname).read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines):
            r = json.loads(line)
            if r["dialect"] != code:
                raise ValueError(f"{fname} row {idx}: dialect {r['dialect']!r} != {code!r}")
            passage = r["flores_passage"].strip()
            if r["link"] in per:
                if per[r["link"]][0] != passage:
                    raise ValueError(f"{fname}: link {r['link']} maps to two passages")
                continue
            per[r["link"]] = (passage, idx)
        out[lang] = per
    return out, files


@dataclass
class FloresSide:
    """FLORES-IN sentences for one Indic language, keyed by normalised English sentence."""

    enxx: dict[str, tuple[int, str]]  # en -> (row in flores_en_xx, xx target)
    xxen: dict[str, tuple[int, str]]  # en -> (row in flores_xx_en, xx source)


def _load_flores(
    fetch: Fetcher, src: Mapping[str, Any], indic: Sequence[str], split: str
) -> dict[str, FloresSide]:
    sides: dict[str, FloresSide] = {}
    for lang in indic:
        enxx: dict[str, tuple[int, str]] = {}
        xxen: dict[str, tuple[int, str]] = {}
        for direction, table in (("enxx", enxx), ("xxen", xxen)):
            a, b = ("en", lang) if direction == "enxx" else (lang, "en")
            fname = f"flores_{a}_{b}_{split}.json"
            for idx, ex in enumerate(_load_examples(fetch(src["repo"], src["revision"], fname))):
                if ex["lang"] != lang or ex["translation_direction"] != direction:
                    raise ValueError(f"{fname} row {idx}: unexpected lang/direction")
                en = normalize_ws(ex["source"] if direction == "enxx" else ex["target"])
                xx = normalize_ws(ex["target"] if direction == "enxx" else ex["source"])
                if en in table:
                    raise ValueError(f"{fname}: duplicate English sentence at row {idx}")
                table[en] = (idx, xx)
        sides[lang] = FloresSide(enxx, xxen)
    return sides


def classify_passages(
    passages_en: Mapping[str, str], devtest_en: Iterable[str], dev_en: Iterable[str]
) -> dict[str, tuple[str, list[str]]]:
    """link -> (origin, devtest sentences in passage order).

    origin is "devtest" (only devtest sentences found), "dev" (only dev sentences found),
    "mixed" or "none".
    """
    devtest = sorted(set(devtest_en))
    dev = sorted(set(dev_en))
    out: dict[str, tuple[str, list[str]]] = {}
    for link in sorted(passages_en):
        p = normalize_ws(passages_en[link])
        t_hits = locate_sentences(p, devtest)
        d_hits = [s for s in dev if s in p]
        origin = {
            (True, False): "devtest",
            (False, True): "dev",
            (True, True): "mixed",
            (False, False): "none",
        }[(bool(t_hits), bool(d_hits))]
        out[link] = (origin, t_hits)
    return out


def build_flores_and_belebele(
    cfg: Mapping[str, Any], fetch: Fetcher
) -> tuple[TaskBuild, TaskBuild, TaskBuild]:
    """translation, translation_xx2en (FLORES devtest) and summarisation (Belebele, dev only).

    Flores-IN files carry no row ids and each file has its own order, so sentences are aligned
    across languages by their (whitespace-normalised) English side, which is unique and identical
    across the five language pairs. Translation items group the devtest sentences of one
    Belebele passage (coherent multi-sentence text, D-013). Summarisation uses only Belebele
    passages built from FLORES *dev*, so it shares no sentence with the translation tasks.
    """
    langs: list[str] = cfg["languages"]
    indic: list[str] = cfg["indic_languages"]
    names: dict[str, str] = cfg["language_names"]
    seed = cfg["seed"]
    fsrc = cfg["sources"]["flores_in"]
    bsrc = cfg["sources"]["belebele"]

    devtest = _load_flores(fetch, fsrc, indic, "test")
    dev_en = set(
        normalize_ws(ex["source"])
        for ex in _load_examples(fetch(fsrc["repo"], fsrc["revision"], "flores_en_hi_dev.json"))
    )
    en_sets = [set(s.enxx) for s in devtest.values()] + [set(s.xxen) for s in devtest.values()]
    common_en = set.intersection(*en_sets)
    stats_fl: dict[str, Any] = {
        "devtest_sentences_per_file": sorted({len(s) for s in en_sets}),
        "devtest_sentences_common": len(common_en),
        # enxx target vs xxen source for the same sentence, after whitespace normalisation
        "direction_text_mismatch": {
            lang: sum(
                1 for en in common_en if devtest[lang].enxx[en][1] != devtest[lang].xxen[en][1]
            )
            for lang in indic
        },
    }

    bel, bel_files = _belebele_passages(fetch, bsrc, langs)
    common_links = set.intersection(*(set(bel[lang]) for lang in langs))
    origin = classify_passages({k: bel["en"][k][0] for k in common_links}, common_en, dev_en)
    origin_counts = Counter(o for o, _ in origin.values())

    # Each devtest sentence must belong to at most one translation item.
    owner: dict[str, str] = {}
    shared_sentences = 0
    for link, (o, sents) in sorted(origin.items()):
        if o != "devtest":
            continue
        for s in sents:
            if s in owner:
                shared_sentences += 1
            owner.setdefault(s, link)

    tr_items: dict[str, dict[str, Cell]] = {}
    xx_items: dict[str, dict[str, Cell]] = {}
    tr_links = sorted(link for link, (o, _) in origin.items() if o == "devtest")
    item_of_link = {link: f"flores_devtest:{short_hash(link)}" for link in tr_links}
    rotation = {
        item: indic[k % len(indic)]
        for k, item in enumerate(seeded_rank(item_of_link.values(), seed, "en_source_lang"))
    }
    tmpl_tr = cfg["tasks"]["translation"]["template"]
    tmpl_xx = cfg["tasks"]["translation_xx2en"]["template"]
    for link in tr_links:
        sents = [s for s in origin[link][1] if owner[s] == link]
        if not sents:
            continue
        item_id = item_of_link[link]
        en_text = " ".join(sents)
        cells: dict[str, Cell] = {}
        xcells: dict[str, Cell] = {}
        for lang in indic:
            side = devtest[lang]
            tgt = " ".join(side.enxx[s][1] for s in sents)
            cells[lang] = Cell(
                prompt=render(tmpl_tr, src_name="English", tgt_name=names[lang], src_text=en_text),
                reference_text=tgt,
                source_row_ref={
                    "file": f"flores_en_{lang}_test.json",
                    "rows": [side.enxx[s][0] for s in sents],
                    "belebele_link": link,
                },
                meta={"src_lang": "en", "tgt_lang": lang, "n_sentences": len(sents)},
            )
            src_xx = " ".join(side.xxen[s][1] for s in sents)
            xcells[lang] = Cell(
                prompt=render(tmpl_xx, src_name=names[lang], tgt_name="English", src_text=src_xx),
                reference_text=src_xx,
                source_row_ref={
                    "file": f"flores_{lang}_en_test.json",
                    "rows": [side.xxen[s][0] for s in sents],
                    "belebele_link": link,
                },
                meta={"src_lang": lang, "tgt_lang": "en", "n_sentences": len(sents)},
            )
        src_lang = rotation[item_id]
        side = devtest[src_lang]
        src_text = " ".join(side.xxen[s][1] for s in sents)
        cells["en"] = Cell(
            prompt=render(tmpl_tr, src_name=names[src_lang], tgt_name="English", src_text=src_text),
            reference_text=en_text,
            source_row_ref={
                "file": f"flores_{src_lang}_en_test.json",
                "rows": [side.xxen[s][0] for s in sents],
                "belebele_link": link,
            },
            meta={"src_lang": src_lang, "tgt_lang": "en", "n_sentences": len(sents)},
        )
        tr_items[item_id] = cells
        xx_items[item_id] = xcells

    used = {s for link in tr_links for s in origin[link][1] if owner[s] == link}
    stats_fl.update(
        {
            "belebele_passages_by_flores_split": dict(sorted(origin_counts.items())),
            "devtest_sentences_used": len(used),
            "devtest_sentences_unused_not_in_belebele": len(common_en - used),
            "sentences_in_two_passages": shared_sentences,
            "en_source_lang_rotation": dict(sorted(Counter(rotation.values()).items())),
        }
    )

    # Summarisation: Belebele passages from FLORES dev only.
    tmpl_sum = cfg["tasks"]["summarisation"]["template"]
    sum_items: dict[str, dict[str, Cell]] = {}
    for link in sorted(link for link, (o, _) in origin.items() if o == "dev"):
        item_id = f"belebele:{short_hash(link)}"
        cells = {}
        for lang in langs:
            passage, row = bel[lang][link]
            cells[lang] = Cell(
                prompt=render(tmpl_sum, lang_name=names[lang], passage=passage),
                reference_text=passage,
                source_row_ref={"file": bel_files[lang], "row": row, "link": link},
            )
        sum_items[item_id] = cells
    all_links = set.union(*(set(bel[lang]) for lang in langs))
    stats_sum = {
        "belebele_passages_per_lang": {lang: len(bel[lang]) for lang in langs},
        "dropped_not_in_all_langs": len(all_links - common_links),
        "belebele_passages_by_flores_split": dict(sorted(origin_counts.items())),
        "kept_dev_passages": len(sum_items),
    }
    return (
        TaskBuild("translation", "flores_in", tr_items, dict(stats_fl)),
        TaskBuild("translation_xx2en", "flores_in", xx_items, dict(stats_fl)),
        TaskBuild("summarisation", "belebele", sum_items, stats_sum),
    )


# --------------------------------------------------------------------------- XQuAD-IN


def build_qa(cfg: Mapping[str, Any], fetch: Fetcher) -> TaskBuild:
    """One question per XQuAD passage (seeded choice); passages grouped by English context."""
    langs: list[str] = cfg["languages"]
    names: dict[str, str] = cfg["language_names"]
    src = cfg["sources"]["xquad_in"]
    tmpl = cfg["tasks"]["qa"]["template"]
    rows: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {}
    for lang in langs:
        fname = f"xquad_{lang}_test.json"
        per: dict[str, tuple[int, dict[str, Any]]] = {}
        for idx, ex in enumerate(_load_examples(fetch(src["repo"], src["revision"], fname))):
            if ex["id"] in per:
                raise ValueError(f"{fname}: duplicate id {ex['id']}")
            per[ex["id"]] = (idx, ex)
        rows[lang] = per
    common = set.intersection(*(set(rows[lang]) for lang in langs))
    union = set.union(*(set(rows[lang]) for lang in langs))
    groups: dict[str, list[str]] = defaultdict(list)
    for qid in sorted(common):
        groups[rows["en"][qid][1]["context"]].append(qid)
    items: dict[str, dict[str, Cell]] = {}
    for qids in groups.values():
        qid = seeded_rank(qids, cfg["seed"], "xquad_question")[0]
        item_id = f"xquad_test:{qid}"
        cells: dict[str, Cell] = {}
        for lang in langs:
            idx, ex = rows[lang][qid]
            context, question = ex["context"].strip(), ex["question"].strip()
            cells[lang] = Cell(
                prompt=render(tmpl, lang_name=names[lang], context=context, question=question),
                reference_text=f"{context}\n{question}",
                source_row_ref={"file": f"xquad_{lang}_test.json", "row": idx, "id": qid},
                meta={"gold_answers": [a["text"] for a in ex["answers"]], "title": ex.get("title")},
            )
        items[item_id] = cells
    stats = {
        "questions_per_lang": {lang: len(rows[lang]) for lang in langs},
        "dropped_not_in_all_langs": len(union - common),
        "passages": len(groups),
        "questions_per_item": 1,
    }
    return TaskBuild("qa", "xquad_in", items, stats)


# --------------------------------------------------------------------------- Aya Dolly-MT


def build_chat(cfg: Mapping[str, Any], fetch: Fetcher) -> TaskBuild:
    import pyarrow.parquet as pq

    langs: list[str] = cfg["languages"]
    src = cfg["sources"]["aya_dolly_mt"]
    codes: dict[str, str] = src["lang_codes"]
    code_to_lang = {v: k for k, v in codes.items()}
    table = pq.read_table(fetch(src["repo"], src["revision"], src["file"])).to_pylist()
    rows: dict[str, dict[int, tuple[int, dict[str, Any]]]] = {lang: {} for lang in langs}
    for idx, r in enumerate(table):
        lang = code_to_lang.get(r["language"])
        if lang is None:
            continue
        if r["id"] in rows[lang]:
            raise ValueError(f"aya {r['language']}: duplicate id {r['id']}")
        rows[lang][r["id"]] = (idx, r)
    common = set.intersection(*(set(rows[lang]) for lang in langs))
    union = set.union(*(set(rows[lang]) for lang in langs))
    items: dict[str, dict[str, Cell]] = {}
    for rid in sorted(common):
        item_id = f"dolly_mt:{rid:03d}"
        source_ids = {rows[lang][rid][1]["source_id"] for lang in langs}
        if len(source_ids) != 1:
            raise ValueError(f"aya id {rid}: source_id differs across languages")
        cells: dict[str, Cell] = {}
        for lang in langs:
            idx, r = rows[lang][rid]
            text = r["inputs"].strip()
            cells[lang] = Cell(
                prompt=render(cfg["tasks"]["chat"]["template"], inputs=text),
                reference_text=text,
                source_row_ref={"file": src["file"], "row": idx, "id": rid},
                meta={"dolly_source_id": r["source_id"], "aya_language": r["language"]},
                translationese=lang != "en",
            )
        items[item_id] = cells
    stats = {
        "rows_per_lang": {lang: len(rows[lang]) for lang in langs},
        "dropped_not_in_all_langs": len(union - common),
    }
    return TaskBuild("chat", "aya_dolly_mt", items, stats)


# --------------------------------------------------------------------------- held-out (TODO)


def build_heldout_in22_conv(cfg: Mapping[str, Any], fetch: Fetcher) -> TaskBuild | None:
    """TODO(owner): IN22-Conv held-out domain (split ``heldout``).

    ``ai4bharat/IN22-Conv`` is gated (accept terms + HF token). Not built until the owner
    provides ``HF_TOKEN`` in a gitignored ``.env`` and sets ``sources.in22_conv.enabled``.
    Planned design: docs/research/data_sources.md section 2.2.
    """
    if not cfg["sources"]["in22_conv"].get("enabled", False):
        return None
    raise NotImplementedError("IN22-Conv builder pending owner's HF token (see D-013)")


# --------------------------------------------------------------------------- tokenisation


class ChatTokenizer:
    """Qwen3 tokenizer at the pinned revision, with its chat template."""

    def __init__(self, repo: str, revision: str, template_kwargs: Mapping[str, Any]) -> None:
        from transformers import AutoTokenizer

        self.repo, self.revision = repo, revision
        self.template_kwargs = dict(template_kwargs)
        self.tok: Any = AutoTokenizer.from_pretrained(repo, revision=revision)

    def chat_text(self, messages: list[dict[str, str]]) -> str:
        out = self.tok.apply_chat_template(messages, tokenize=False, **self.template_kwargs)
        if not isinstance(out, str):
            raise TypeError("apply_chat_template(tokenize=False) did not return a string")
        return out

    def encode(self, text: str) -> list[int]:
        return [int(t) for t in self.tok(text, add_special_tokens=False)["input_ids"]]

    def decode(self, ids: Sequence[int]) -> str:
        return str(self.tok.decode(list(ids), skip_special_tokens=False))

    def info(self) -> dict[str, Any]:
        import tokenizers
        import transformers

        from indicspec.tokens.vocab import PinnedTokenizer, fetch_tokenizer_json

        tj = fetch_tokenizer_json(PinnedTokenizer(self.repo, self.revision))
        return {
            "repo": self.repo,
            "revision": self.revision,
            "tokenizer_json_sha256": sha256_file(tj),
            "chat_template_sha256": sha256_bytes(str(self.tok.chat_template).encode("utf-8")),
            "chat_template_kwargs": self.template_kwargs,
            "transformers_version": transformers.__version__,
            "tokenizers_version": tokenizers.__version__,
        }


def load_chat_tokenizer(cfg: Mapping[str, Any]) -> ChatTokenizer:
    models = yaml.safe_load(MODELS_CONFIG.read_text(encoding="utf-8"))
    target = models["targets"][cfg["tokenizer"]["model_key"]]
    tok = ChatTokenizer(
        target["repo"], target["revision"], cfg["tokenizer"]["chat_template_kwargs"]
    )
    prefix = models.get("tokenizer_sha256_prefix")
    if prefix and not tok.info()["tokenizer_json_sha256"].startswith(prefix):
        raise RuntimeError("tokenizer.json sha256 does not match configs/models.yaml")
    return tok


# --------------------------------------------------------------------------- assembly


def make_rows(
    tb: TaskBuild, cfg: Mapping[str, Any], tok: ChatTokenizer
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_cfg = cfg["tasks"][tb.task]
    src = cfg["sources"][tb.source_key]
    langs: list[str] = task_cfg["langs"]
    sp = cfg["splits"]
    for item_id, cells in tb.items.items():
        if sorted(cells) != sorted(langs):
            raise ValueError(f"{tb.task} {item_id}: languages {sorted(cells)} != {sorted(langs)}")
    res = assign_splits(
        tb.items, cfg["seed"], sp["pilot_frac"], sp["tuning_frac"], sp["report_cap"]
    )
    rows: list[dict[str, Any]] = []
    nfc_changed = 0
    for item_id in sorted(res.assignment):
        for lang in sorted(tb.items[item_id]):
            cell = tb.items[item_id][lang]
            # The Qwen3 tokenizer.json normalizer is NFC, so token ids always encode NFC text.
            # Store NFC text so n_chars counts exactly what is tokenised and decode(ids) == text.
            prompt, reference = nfc(cell.prompt), nfc(cell.reference_text)
            nfc_changed += (prompt != cell.prompt) or (reference != cell.reference_text)
            messages = [{"role": "user", "content": prompt}]
            prompt_ids = tok.encode(tok.chat_text(messages))
            ref_ids = tok.encode(reference)
            rows.append(
                {
                    "item_id": item_id,
                    "task": tb.task,
                    "lang": lang,
                    "split": res.assignment[item_id],
                    "source_repo": src["repo"],
                    "source_revision": src["revision"],
                    "source_row_ref": cell.source_row_ref,
                    "licence": src["licence"],
                    "translationese": bool(task_cfg.get("translationese", False))
                    or cell.translationese,
                    "flores_derived": bool(task_cfg.get("flores_derived", False)),
                    "messages": messages,
                    "reference_text": reference,
                    **text_stats(reference),
                    "prompt_token_ids": prompt_ids,
                    "n_prompt_tokens": len(prompt_ids),
                    "ref_token_ids": ref_ids,
                    "n_ref_tokens": len(ref_ids),
                    "meta": cell.meta,
                }
            )
    stats = dict(tb.stats)
    stats["items_total_before_split"] = len(tb.items)
    stats["dropped_by_report_cap"] = len(res.dropped_by_cap)
    stats["rows_changed_by_nfc"] = nfc_changed
    return rows, stats


def file_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_lang: dict[str, Counter[str]] = defaultdict(Counter)
    items: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_lang[r["lang"]][r["split"]] += 1
        items[r["split"]].add(r["item_id"])
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "items_per_split": {s: len(items[s]) for s in SPLITS if s in items},
        "rows_per_lang_split": {
            lang: {s: c[s] for s in SPLITS if s in c} for lang, c in sorted(by_lang.items())
        },
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json_line(r))


def build(
    cfg: Mapping[str, Any], bench_dir: Path, raw_dir: Path, log: Any = print
) -> dict[str, Any]:
    """Build every task into ``bench_dir`` and return the manifest dict."""
    fetch = Fetcher(raw_dir)
    licences = {}
    for key, src in cfg["sources"].items():
        if src.get("revision"):
            licences[key] = check_licence(fetch, src)
            log(f"[data] {src['repo']}@{src['revision'][:10]} licence={licences[key]} (README)")
    tok = load_chat_tokenizer(cfg)
    tr, xx, summ = build_flores_and_belebele(cfg, fetch)
    builds = [tr, xx, build_qa(cfg, fetch), summ, build_chat(cfg, fetch)]
    heldout = build_heldout_in22_conv(cfg, fetch)
    if heldout is not None:  # pragma: no cover - gated, TODO
        builds.append(heldout)

    files: dict[str, Any] = {}
    task_stats: dict[str, Any] = {}
    for tb in builds:
        rows, stats = make_rows(tb, cfg, tok)
        path = bench_dir / f"{tb.task}.jsonl"
        write_jsonl(path, rows)
        files[f"bench/{tb.task}.jsonl"] = file_summary(path, rows)
        task_stats[tb.task] = {
            "source": tb.source_key,
            "langs": cfg["tasks"][tb.task]["langs"],
            **stats,
        }
        log(f"[data] {tb.task}: {len(rows)} rows -> {path}")

    sources = {}
    for key, src in cfg["sources"].items():
        entry = {
            k: src[k]
            for k in ("repo", "revision", "config", "url", "licence", "licence_verified_at")
            if k in src
        }
        if src.get("revision"):
            entry["files_sha256"] = dict(sorted(fetch.fetched[src["repo"]].items()))
        else:
            entry["status"] = "TODO: gated, not downloaded (needs owner's HF token)"
        sources[key] = entry
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "builder": "python -m indicspec.data.build",
        "seed": cfg["seed"],
        "splits": cfg["splits"],
        "generation_variants": cfg["generation_variants"],
        "prompt_instruction_language": cfg["prompt_instruction_language"],
        "tokenizer": tok.info(),
        "sources": sources,
        "tasks": task_stats,
        "files": files,
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def compare_manifests(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    """Differences in output-file checksums/row counts and raw-source checksums."""
    problems: list[str] = []
    ef, af = expected.get("files", {}), actual.get("files", {})
    for name in sorted(set(ef) | set(af)):
        if name not in af:
            problems.append(f"{name}: missing from rebuild")
        elif name not in ef:
            problems.append(f"{name}: not in MANIFEST")
        else:
            for k in ("sha256", "rows"):
                if ef[name][k] != af[name][k]:
                    problems.append(f"{name}: {k} {ef[name][k]} != rebuilt {af[name][k]}")
    for key, src in expected.get("sources", {}).items():
        got = actual.get("sources", {}).get(key, {})
        for fname, sha in src.get("files_sha256", {}).items():
            if got.get("files_sha256", {}).get(fname) != sha:
                problems.append(f"source {key}/{fname}: sha256 differs from MANIFEST")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="rebuild into a temporary directory, compare sha256 with MANIFEST, then install "
        "the files into --bench-dir; never rewrites MANIFEST",
    )
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    def log(msg: str) -> None:
        print(msg, file=sys.stderr)

    if not args.verify:
        manifest = build(cfg, args.bench_dir, args.raw_dir, log)
        write_manifest(args.manifest, manifest)
        log(f"[data] wrote {args.manifest}")
        return 0

    if not args.manifest.exists():
        log(f"[data] --verify: {args.manifest} does not exist; run without --verify first")
        return 2
    expected = json.loads(args.manifest.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="indicspec-bench-") as tmp:
        actual = build(cfg, Path(tmp), args.raw_dir, log)
        problems = compare_manifests(expected, actual)
        if problems:
            for p in problems:
                log(f"[data] VERIFY FAIL {p}")
            return 1
        args.bench_dir.mkdir(parents=True, exist_ok=True)
        for name in actual["files"]:
            src_path = Path(tmp) / Path(name).name
            (args.bench_dir / Path(name).name).write_bytes(src_path.read_bytes())
    log(f"[data] verify OK: {len(actual['files'])} files match {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
