"""Summary table of the exploratory cross-model quick test (NOT pre-registered; D-025).

Reads results/raw/quick_models/{tokens,speed,speed_hf,quality}.jsonl and writes
results/tables/crossmodel_summary.csv: one row per (kept model, language) with tokens per word,
n-gram speed-up (k=8) at one user, words per second without/with the drafter, the repetition score,
and answer quality (QA accuracy, chrF, script share, truncation).

Kept models and why the rest are excluded: see D-025. Lumma has no vLLM support (factorised output
head), so its speed comes from transformers, with Qwen3-0.6B run through the same path as reference.
Speed rows from the first, interrupted Tamil-LLaMA run (before 22:05) are dropped.

Usage: python scripts/crossmodel_summary.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from quick_quality import SCORED_TASKS, chrf, chrf_stats

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results/raw/quick_models"
LANGS = ["en", "hi", "ta", "te", "kn", "ml"]
# name -> (speed source, model key in that source, tokenizer record name)
KEEP = {
    "qwen3-4b": ("vllm", "qwen3-4b", "qwen3-4b"),
    "qwen3-0.6b": ("vllm", "qwen3-0.6b", "qwen3-4b"),
    "gemma-4-e4b": ("vllm", "gemma-4-e4b", "gemma-4-e4b"),
    "sarvam-1": ("vllm", "sarvam-1", "sarvam-1"),
    "sarvam-1 (fixed length)": ("vllm", "sarvam-1+fixedlen", "sarvam-1"),
    "navarasa-7b": ("vllm", "navarasa-7b", "navarasa-7b"),
    "tamil-llama-7b": ("vllm", "tamil-llama-7b", "tamil-llama-7b"),
    "openhathi-7b": ("vllm", "openhathi-7b", "openhathi-7b"),
    "qwen3-0.6b [transformers]": ("hf", "Qwen3-0.6B", "qwen3-4b"),
    "lumma-0.6b [transformers]": ("hf", "Lumma-0.6B-Instruct", "lumma-0.6b"),
}


def jsonl(name: str) -> list[dict]:
    p = RAW / name
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    tok = {r["model"]: r for r in jsonl("tokens.jsonl")}  # last record per name wins
    speed = [
        r
        for r in jsonl("speed.jsonl")
        if not (r["model"] == "tamil-llama-7b" and r["time"] < "2026-11-27T22:05")
    ]
    sp: dict[tuple, list[float]] = defaultdict(list)
    rp: dict[tuple, list[float]] = defaultdict(list)
    for src, rows in (("vllm", speed), ("hf", jsonl("speed_hf.jsonl"))):
        for r in rows:
            sp[(src, r["model"], r["arm"], r["lang"])].append(r["output_tokens"] / r["latency_s"])
            if r["arm"] == "A0":
                rp[(src, r["model"], r["lang"])].append(r["rep4"])
    refs = {}
    for task in SCORED_TASKS:
        for x in (REPO / f"data/bench/{task}.jsonl").read_text(encoding="utf-8").splitlines():
            b = json.loads(x)
            if b["split"] == "tuning":
                refs[(b["lang"], f"{task}:{b['item_id']}")] = b["reference_text"]
    qual: dict[tuple, list[dict]] = defaultdict(list)
    for r in jsonl("quality.jsonl") if (RAW / "quality.jsonl").exists() else []:
        qual[(r["model"], r["lang"])].append(r)
    out = []
    for name, (src, key, tkey) in KEEP.items():
        tpw = tok[tkey]["tok_per_word"]
        qname = {"lumma-0.6b [transformers]": "lumma-0.6b"}.get(name, name)
        for lang in LANGS:
            a0, ng = sp.get((src, key, "A0", lang)), sp.get((src, key, "ngram8", lang))
            if not a0 or not ng:
                continue
            qs = (
                qual.get((qname, lang), [])
                if "fixed" not in name and "qwen3-0.6b [" not in name
                else []
            )
            qa = [r["qa_hit"] for r in qs if "qa_hit" in r]
            stats = [
                chrf_stats(r["text"], refs[(lang, r["item"])])
                for r in qs
                if r["task"] in SCORED_TASKS and (lang, r["item"]) in refs
            ]
            out.append(
                {
                    "model": name,
                    "engine": "transformers" if src == "hf" else "vllm",
                    "language": lang,
                    "tokens_per_word": tpw[lang],
                    "vocab": tok[tkey]["vocab"],
                    "ngram_speedup": st.mean(ng) / st.mean(a0),
                    "words_per_s_no_spec": st.mean(a0) / tpw[lang],
                    "words_per_s_ngram": st.mean(ng) / tpw[lang],
                    "rep4": st.mean(rp[(src, key, lang)]),
                    "qa_acc": st.mean(qa) if qa else None,
                    "n_qa": len(qa),
                    "chrf": chrf(stats) if stats else None,
                    "script_share": st.mean(r["script"] for r in qs) if qs else None,
                    "truncated": st.mean(r["truncated"] for r in qs) if qs else None,
                }
            )
    dest = REPO / "results/tables/crossmodel_summary.csv"
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {dest.relative_to(REPO)} ({len(out)} rows)")
    # Tokenizer table: every tokenizer measured that belongs to a kept model, plus Sarvam-105B
    # (a tokenizer-only reference; same tokenizer as Sarvam-30B).
    tok_keep = {
        "qwen3-4b": "Qwen3 (all sizes)",
        "gemma-4-e4b": "Gemma 4",
        "sarvam-105b": "Sarvam-105B / 30B",
        "sarvam-1": "Sarvam-1",
        "navarasa-7b": "Gemma 1 (Navarasa)",
        "tamil-llama-7b": "Tamil-LLaMA",
        "openhathi-7b": "OpenHathi (Llama-2 + Hindi)",
        "lumma-0.6b": "FrontiersMind Lumma / Nandi",
    }
    dest2 = REPO / "results/tables/crossmodel_tokens.csv"
    with open(dest2, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tokenizer", "model_key", "vocab", *LANGS, "dravidian_mean"])
        for k, label in tok_keep.items():
            t = tok[k]["tok_per_word"]
            drav = st.mean(t[x] for x in ("ta", "te", "kn", "ml"))
            w.writerow([label, k, tok[k]["vocab"], *(f"{t[x]:.4f}" for x in LANGS), f"{drav:.4f}"])
    print(f"wrote {dest2.relative_to(REPO)}")


if __name__ == "__main__":
    main()
