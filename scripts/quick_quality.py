"""Exploratory answer-quality check for the cross-model quick test (NOT pre-registered).

Same tuning-split items as scripts/quick_models.py (first 10 per language that fit), greedy, up to
1,024 new tokens so that fragment-heavy tokenizers are not cut off. Scores, per model x language:
  qa_acc   QA only: a gold answer string (IndicGenBench xquad) appears in the output (normalised)
  chrf     chrF (char 6-grams, beta 2; sacrebleu-style corpus statistics) vs the reference text,
           over QA + summarisation + translation (chat references are too short to be meaningful)
  script   share of output letters in the expected script (Latin for en)
  trunc    share of outputs that hit the token limit
vLLM serves models it supports; custom-code models (EXTRA_ARGS in quick_models.py) run in
transformers. Greedy output does not depend on speculation, so quality is measured once
(no drafter).

Usage (WSL, venv-vllm, GPU free): python scripts/quick_quality.py MODEL [MODEL ...]
Outputs: results/raw/quick_models/quality.jsonl (one row per output, with text) + printed summary.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent))
import quick_models as q

SCRIPT = {
    "en": (0x0041, 0x007A),
    "hi": (0x0900, 0x097F),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
}
SCORED_TASKS = {"qa", "summarisation", "translation"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    return re.sub(r"[\s\W_]+", "", s)


def script_share(text: str, lang: str) -> float:
    letters = [c for c in text if c.isalpha() or unicodedata.category(c).startswith("M")]
    if not letters:
        return 0.0
    lo, hi = SCRIPT[lang]
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters)


def chrf_stats(hyp: str, ref: str, n: int = 6) -> list[tuple[int, int, int]]:
    h, r = hyp.replace(" ", ""), ref.replace(" ", "")
    out = []
    for k in range(1, n + 1):
        hc = Counter(h[i : i + k] for i in range(len(h) - k + 1))
        rc = Counter(r[i : i + k] for i in range(len(r) - k + 1))
        out.append((sum((hc & rc).values()), sum(hc.values()), sum(rc.values())))
    return out


def chrf(stats: list[list[tuple[int, int, int]]], beta: float = 2.0) -> float:
    ps, rs = [], []
    for k in range(6):
        m = sum(s[k][0] for s in stats)
        hyp = sum(s[k][1] for s in stats)
        ref = sum(s[k][2] for s in stats)
        ps.append(m / hyp if hyp else 0.0)
        rs.append(m / ref if ref else 0.0)
    p, r = st.mean(ps), st.mean(rs)
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r) if p + r else 0.0


def gen_vllm(path: str, name: str, enc: list[list[int]], max_tokens: int) -> list[tuple[str, int]]:
    p = q.serve(path, None, q.OUT / f"server-{name}-quality.log")
    url = f"http://127.0.0.1:{q.PORT}/v1/completions"

    def one(ids: list[int]) -> tuple[str, int]:
        body = {"model": "m", "prompt": ids, "max_tokens": max_tokens, "temperature": 0}
        r = httpx.post(url, json=body, timeout=900).json()
        return r["choices"][0]["text"], r["usage"]["completion_tokens"]

    try:
        with ThreadPoolExecutor(8) as ex:
            return list(ex.map(one, enc))
    finally:
        q.stop(p)


def gen_hf(path: str, repo: str, tok: Any, enc: list[list[int]], max_tokens: int) -> list:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to("cuda").eval()
    raw = [tok.eos_token_id, getattr(model.generation_config, "eos_token_id", None)]
    eos = sorted({e for x in raw for e in (x if isinstance(x, list) else [x]) if e is not None})
    out = []
    with torch.inference_mode():
        for ids in enc:
            g = model.generate(
                torch.tensor([ids], device="cuda"),
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=eos or None,
                pad_token_id=tok.pad_token_id or (eos[0] if eos else 0),
            )
            new = g[0, len(ids) :].tolist()
            out.append((tok.decode(new, skip_special_tokens=True), len(new)))
    del model
    torch.cuda.empty_cache()
    return out


def run(name: str, n: int, max_tokens: int) -> dict[str, dict[str, float]]:
    from transformers import AutoTokenizer

    repo = q.MODELS[name][0]
    path = q.local(repo)
    custom = repo in q.EXTRA_ARGS
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=custom)
    prompts = q.pick(q.bench(), 2 * n)
    lim = q.max_len(path) - max_tokens
    enc = {x: [q.encode_prompt(tok, p["messages"]) for p in ps] for x, ps in prompts.items()}
    keep = [i for i in range(len(prompts["en"])) if all(len(enc[x][i]) <= lim for x in q.LANGS)]
    keep = keep[:n]
    items = [(x, prompts[x][i], enc[x][i]) for x in q.LANGS for i in keep]
    flat = [e for _, _, e in items]
    t0 = time.time()
    outs = (
        gen_hf(path, repo, tok, flat, max_tokens)
        if custom
        else gen_vllm(path, name, flat, max_tokens)
    )
    res: dict[str, dict[str, float]] = {}
    for lang in q.LANGS:
        rows = [(it, o) for (x, it, _), o in zip(items, outs, strict=True) if x == lang]
        qa = [it for it, _ in rows if it["task"] == "qa"]
        qa_hits = []
        stats = []
        for it, (text, k) in rows:
            rec = {
                "model": name,
                "repo": repo,
                "lang": lang,
                "item": f"{it['task']}:{it['item_id']}",
                "task": it["task"],
                "output_tokens": k,
                "truncated": k >= max_tokens,
                "script": script_share(text, lang),
                "text": text,
                "max_tokens": max_tokens,
            }
            if it["task"] == "qa":
                gold = (it.get("meta") or {}).get("gold_answers") or []
                rec["qa_hit"] = any(norm(g) and norm(g) in norm(text) for g in gold)
                qa_hits.append(rec["qa_hit"])
            if it["task"] in SCORED_TASKS:
                stats.append(chrf_stats(text, it["reference_text"]))
            with open(q.OUT / "quality.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        res[lang] = {
            "qa_acc": st.mean(qa_hits) if qa_hits else float("nan"),
            "n_qa": len(qa),
            "chrf": chrf(stats),
            "script": st.mean(script_share(t, lang) for _, (t, _) in rows),
            "trunc": st.mean(k >= max_tokens for _, (_, k) in rows),
        }
    print(f"\n{name}: n={len(keep)}/language, max_tokens={max_tokens}, {time.time() - t0:.0f}s")
    print("lang  qa_acc  chrF  script  trunc")
    for lang, v in res.items():
        print(
            f"{lang:5} {v['qa_acc']:5.2f} {v['chrf']:6.1f} {v['script']:6.2f} {v['trunc']:6.2f}"
            f"   (qa n={v['n_qa']})"
        )
    sys.stdout.flush()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()
    for name in a.models:
        try:
            run(name, a.n, a.max_tokens)
        except Exception as e:
            print(f"\nFAILED {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            import subprocess

            subprocess.run(["pkill", "-f", "vllm serve"], check=False)
            time.sleep(10)


if __name__ == "__main__":
    main()
