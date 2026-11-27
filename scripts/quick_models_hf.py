"""Exploratory quick test in Hugging Face transformers, for models vLLM cannot run (custom code).

Same prompts and measures as scripts/quick_models.py (tuning split, first 10 items per language that
fit, greedy, 128 new tokens, one request at a time), but the n-gram drafter is transformers'
prompt-lookup decoding (draft 8 tokens, match n-grams up to 5). Absolute speeds are not comparable
with vLLM, so reference models are run through this same path. Acceptance length tau is measured
as new tokens per target forward pass (a hook counts forward calls).

Usage (WSL, venv-vllm, GPU free):
  python scripts/quick_models_hf.py MODEL_REPO [MODEL_REPO ...] [--n 10] [--max-tokens 128]
Custom model code runs with trust_remote_code only for repos listed in TRUSTED (code reviewed).
Outputs: results/raw/quick_models/speed_hf.jsonl (append) and a printed summary.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import quick_models as q

# Reviewed 2026-11-27: modeling_lumma.py / configuration_lumma.py import only torch + transformers.
TRUSTED = {"FrontiersMind/Lumma-0.6B-Instruct", "FrontiersMind/Nandi-Mini-150M-Instruct"}


def run(repo: str, n: int, max_tokens: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = q.local(repo)
    trust = repo in TRUSTED
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, trust_remote_code=trust
    ).to("cuda")
    model.eval()
    calls = {"n": 0}
    model.register_forward_pre_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
    prompts = q.pick(q.bench(), 2 * n)
    lim = q.max_len(path) - max_tokens
    enc = {x: [q.encode_prompt(tok, p["messages"]) for p in ps] for x, ps in prompts.items()}
    keep = [i for i in range(len(prompts["en"])) if all(len(enc[x][i]) <= lim for x in q.LANGS)]
    keep = keep[:n]
    raw = [tok.eos_token_id, getattr(model.generation_config, "eos_token_id", None)]
    eos = sorted({e for x in raw for e in (x if isinstance(x, list) else [x]) if e is not None})
    arms = {"A0": {}, "ngram8": {"prompt_lookup_num_tokens": 8, "max_matching_ngram_size": 5}}
    summary = {}
    with torch.inference_mode():
        warm = torch.tensor([enc["en"][keep[0]]], device="cuda")
        model.generate(warm, max_new_tokens=16, do_sample=False, use_cache=True)
        for arm, extra in arms.items():
            for lang in q.LANGS:
                tps, taus, reps, ntok = [], [], [], []
                for i in keep:
                    ids = torch.tensor([enc[lang][i]], device="cuda")
                    calls["n"] = 0
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    out = model.generate(
                        ids,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=eos or None,
                        pad_token_id=tok.pad_token_id or (eos[0] if eos else 0),
                        **extra,
                    )
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t
                    new = out[0, ids.shape[1] :].tolist()
                    k = len(new)
                    tps.append(k / dt)
                    taus.append(k / max(calls["n"] - 1, 1))  # first call is the prefill
                    reps.append(q.rep4(new))
                    ntok.append(k)
                    rec = {
                        "model": repo.split("/")[-1],
                        "repo": repo,
                        "engine": "transformers",
                        "arm": arm,
                        "lang": lang,
                        "item": f"{prompts[lang][i]['task']}:{prompts[lang][i]['item_id']}",
                        "prompt_tokens": ids.shape[1],
                        "output_tokens": k,
                        "latency_s": dt,
                        "forward_calls": calls["n"],
                        "rep4": q.rep4(new),
                        "max_tokens": max_tokens,
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    with open(q.OUT / "speed_hf.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                summary[(arm, lang)] = (st.mean(tps), st.mean(taus), st.mean(reps), st.mean(ntok))
    print(f"\n{repo} [transformers]; n={len(keep)}/language, max_tokens={max_tokens}")
    print("lang  speedup   tau  tok/s(A0)  rep4(A0)  out_tok(A0)")
    for lang in q.LANGS:
        b, s = summary[("A0", lang)], summary[("ngram8", lang)]
        print(f"{lang:5} {s[0] / b[0]:6.2f}x {s[1]:5.2f} {b[0]:9.1f} {b[2]:9.2f} {b[3]:11.0f}")
    sys.stdout.flush()
    del model
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=128)
    a = ap.parse_args()
    q.OUT.mkdir(parents=True, exist_ok=True)
    for repo in a.repos:
        try:
            run(repo, a.n, a.max_tokens)
        except Exception as e:
            print(f"\nFAILED {repo}: {type(e).__name__}: {str(e)[:400]}", flush=True)


if __name__ == "__main__":
    main()
