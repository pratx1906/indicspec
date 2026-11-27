"""Exploratory cross-model quick test (NOT pre-registered; never used for verdicts).

Question: does the n-gram drafter's language pattern (big Dravidian gain on Qwen3) depend on the
model's tokenizer and training data? For each model: (1) token tax of its tokenizer on the
tuning-split reference texts; (2) one-user speed with and without vLLM's n-gram drafter (k=8,
lookup 2-5, as tuned arm A1.k8) on the same tuning-split prompts, re-tokenised with the model's own
chat format (Alpaca-style if it has none). Also records a repetition score per output (share of
repeated 4-grams), because a looping model makes n-gram drafting look better than it is.

Usage (WSL, venv-vllm, one GPU job at a time):
  python scripts/quick_models.py tokens
  python scripts/quick_models.py speed [MODEL ...] [--n 10] [--max-tokens 128] [--until HH:MM]
Outputs: results/raw/quick_models/{tokens,speed}.jsonl (append) and printed summaries.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO = Path("/mnt/c/Users/Pratyush Dixit/Desktop/Inference")
OUT = REPO / "results/raw/quick_models"
LANGS = ["en", "hi", "ta", "te", "kn", "ml"]
TASKS = ["chat", "qa", "summarisation", "translation"]
VLLM = Path.home() / ".cache/indicspec/venv-vllm/bin/vllm"
MARKS = Path.home() / ".cache/indicspec/quick_models_ready"  # one file per downloaded repo
PORT = 8100
NGRAM = {
    "method": "ngram",
    "num_speculative_tokens": 8,
    "prompt_lookup_min": 2,
    "prompt_lookup_max": 5,
}
ALPACA = "### Instruction:\n{q}\n\n### Response:\n"

# Order = priority: if the time window runs out, models at the end are skipped.
# name: (repo, group, note)
MODELS: dict[str, tuple[str, str, str]] = {
    "qwen3-4b": ("Qwen/Qwen3-4B", "global", "reference; Qwen tokenizer"),
    "gemma-4-e4b": ("google/gemma-4-E4B-it", "global", "Gemma 4, 262k vocabulary"),
    "sarvam-1": ("sarvamai/sarvam-1", "india-scratch", "2.5B, own Indic tokenizer"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "global", "twin of bharat-tiny"),
    "bharat-tiny-1.5b": ("eulogik/Bharat-Tiny-LLM-v2", "india-data", "Qwen2.5-1.5B + Hindi data"),
    "krutrim-1": ("krutrim-ai-labs/Krutrim-1-instruct", "india-scratch", "7B MPT, own tokenizer"),
    "tamil-llama-7b": (
        "abhinand/tamil-llama-7b-instruct-v0.2",
        "india-extended",
        "Llama-2 + Tamil tokens and data",
    ),
    "navarasa-7b": (
        "Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0",
        "india-data",
        "Gemma-7B + Indic instruction data",
    ),
    "malayalam-llama-7b": (
        "abhinand/malayalam-llama-7b-instruct-v0.1",
        "india-extended",
        "Llama-2 + Malayalam tokens and data",
    ),
    "openhathi-7b": (
        "sarvamai/OpenHathi-7B-Hi-v0.1-Base",
        "india-extended",
        "Llama-2 + Hindi tokens (base model)",
    ),
    "gemma-4-e2b": ("google/gemma-4-E2B-it", "global", "Gemma 4, smaller"),
    "pragna-1b": ("soketlabs/pragna-1b", "india-scratch", "Soket AI 1.3B (base)"),
    "ganga-2-1b": ("LingoIITGN/Ganga-2-1B", "india-scratch", "IIT Gandhinagar 1B"),
    "indus-1.2b": ("nickmalhotra/ProjectIndus", "india-scratch", "Tech Mahindra 1.2B, GPT-2"),
    "llama-2-7b": ("meta-llama/Llama-2-7b-hf", "global", "twin of the Llama-2-extended models"),
    "lumma-0.6b": (
        "FrontiersMind/Lumma-0.6B-Instruct",
        "india-scratch",
        "FrontiersMind 0.6B, own 131k tokenizer (vLLM Transformers backend)",
    ),
    "nandi-mini-150m": (
        "FrontiersMind/Nandi-Mini-150M-Instruct",
        "india-scratch",
        "FrontiersMind 150M, same tokenizer (vLLM Transformers backend)",
    ),
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "global", "size-matched reference for Lumma"),
}
# Models with custom code: vLLM's Transformers backend + the repo's own modeling code, reviewed
# 2026-11-27 (imports only torch/transformers).
EXTRA_ARGS: dict[str, list[str]] = {
    "FrontiersMind/Lumma-0.6B-Instruct": ["--model-impl", "transformers", "--trust-remote-code"],
    "FrontiersMind/Nandi-Mini-150M-Instruct": [
        "--model-impl",
        "transformers",
        "--trust-remote-code",
    ],
}
BIG = {
    "gemma-4-e4b",
    "krutrim-1",
    "tamil-llama-7b",
    "navarasa-7b",
    "malayalam-llama-7b",
    "openhathi-7b",
    "llama-2-7b",
}


def local(repo: str) -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:  # cached at a pinned revision without a "main" ref (e.g. Qwen3-4B)
        snaps = sorted(
            (Path.home() / ".cache/huggingface/hub" / f"models--{repo.replace('/', '--')}").glob(
                "snapshots/*"
            )
        )
        if not snaps:
            raise
        return str(snaps[-1])


def ready(repo: str) -> bool:
    return (MARKS / repo.replace("/", "--")).exists()


def bench(split: str = "tuning") -> list[dict[str, Any]]:
    rows = []
    for t in TASKS:
        for line in (REPO / f"data/bench/{t}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["split"] == split and r["lang"] in LANGS:
                rows.append(r)
    return rows


def pick(rows: list[dict[str, Any]], n: int, seed: int = 1234) -> dict[str, list[dict[str, Any]]]:
    """The same n items (task:item_id) in every language, balanced over tasks."""
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["lang"], {})[f"{r['task']}:{r['item_id']}"] = r
    common = sorted(set.intersection(*(set(v) for v in by.values())))
    rng = random.Random(seed)
    per_task = {t: [k for k in common if k.startswith(t + ":")] for t in TASKS}
    for v in per_task.values():
        rng.shuffle(v)
    chosen: list[str] = []
    while len(chosen) < n and any(per_task.values()):
        for t in TASKS:
            if per_task[t] and len(chosen) < n:
                chosen.append(per_task[t].pop())
    return {lang: [by[lang][k] for k in chosen] for lang in LANGS}


def encode_prompt(tok: Any, messages: list[dict[str, str]]) -> list[int]:
    if getattr(tok, "chat_template", None):
        kw = {"add_generation_prompt": True, "tokenize": False}
        try:
            text = tok.apply_chat_template(messages, enable_thinking=False, **kw)
        except TypeError:
            text = tok.apply_chat_template(messages, **kw)
        # the rendered template already contains BOS/special tokens where the model expects them
        return list(tok.encode(text, add_special_tokens=False))
    q = "\n\n".join(m["content"] for m in messages if m["role"] != "assistant")
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    return bos + tok.encode(ALPACA.format(q=q), add_special_tokens=False)


def rep4(ids: list[int]) -> float:
    grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def max_len(path: str) -> int:
    c = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    keys = ("max_position_embeddings", "max_seq_len", "n_positions", "seq_length")
    vals = [d.get(k) for d in (c, c.get("text_config") or {}) for k in keys]
    return min([4096] + [int(v) for v in vals if isinstance(v, int)])


def tokens_cmd() -> None:
    from transformers import AutoTokenizer

    rows = [r for r in bench() if r.get("reference_text")]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'model':20}" + "".join(f"{x:>7}" for x in LANGS) + "   (tokens per word)")
    for name, (repo, group, _) in MODELS.items():
        try:
            tok = AutoTokenizer.from_pretrained(local(repo))
        except Exception as e:
            print(f"{name:20} unavailable: {str(e)[:80]}")
            continue
        res = {}
        for lang in LANGS:
            sel = [r for r in rows if r["lang"] == lang]
            ntok = sum(len(tok.encode(r["reference_text"], add_special_tokens=False)) for r in sel)
            nw = sum(len(r["reference_text"].split()) for r in sel)
            res[lang] = ntok / nw
        rec = {"model": name, "repo": repo, "group": group, "vocab": len(tok), "tok_per_word": res}
        with open(OUT / "tokens.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            f"{name:20}"
            + "".join(f"{res[x]:7.2f}" for x in LANGS)
            + f"   ml/en {res['ml'] / res['en']:.1f}x  vocab {len(tok)}",
            flush=True,
        )


def metrics() -> tuple[float, float]:
    text = httpx.get(f"http://127.0.0.1:{PORT}/metrics", timeout=30).text
    d = a = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total"):
            d += float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            a += float(line.split()[-1])
    return d, a


def serve(path: str, spec: dict[str, Any] | None, log: Path) -> subprocess.Popen:
    cmd = [str(VLLM), "serve", path, "--served-model-name", "m", "--host", "127.0.0.1"]
    cmd += ["--port", str(PORT), "--dtype", "bfloat16", "--max-model-len", str(max_len(path))]
    cmd += ["--gpu-memory-utilization", "0.85", "--seed", "1234", "--max-num-seqs", "4"]
    cmd += next((v for k, v in EXTRA_ARGS.items() if k.replace("/", "--") in path), [])
    if spec:
        cmd += ["--speculative-config", json.dumps(spec)]
    env = {**os.environ, "VLLM_WSL2_ENABLE_PIN_MEMORY": "1", "HF_HUB_OFFLINE": "1"}
    fh = open(log, "a", encoding="utf-8")  # noqa: SIM115
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise RuntimeError(f"server exited ({p.returncode}); see {log}")
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                return p
        except httpx.HTTPError:
            pass
        time.sleep(3)
    stop(p)
    raise RuntimeError("server did not become healthy in 15 min")


def stop(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=90)
    except Exception:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGKILL)


def power_ok() -> bool:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=enforced.power.limit", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        return float(out) >= 140
    except ValueError:
        return True  # not reported under WSL


def one_model(
    name: str,
    prompts: dict[str, list[dict[str, Any]]],
    max_tokens: int,
    n: int,
    fixed_len: bool = False,
) -> None:
    from transformers import AutoTokenizer

    repo = MODELS[name][0]
    path = local(repo)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=repo in EXTRA_ARGS)
    lim = max_len(path) - max_tokens
    enc = {lang: [encode_prompt(tok, p["messages"]) for p in ps] for lang, ps in prompts.items()}
    keep = [i for i in range(len(prompts["en"])) if all(len(enc[x][i]) <= lim for x in LANGS)]
    keep = keep[:n]  # first n candidate items that fit this model's context in every language
    prompts = {x: [prompts[x][i] for i in keep] for x in LANGS}
    enc = {x: [enc[x][i] for i in keep] for x in LANGS}
    summary: dict[tuple[str, str], tuple[float, float | None, float, float]] = {}
    url = f"http://127.0.0.1:{PORT}/v1/completions"
    for arm, spec in (("A0", None), ("ngram8", NGRAM)):
        p = serve(path, spec, OUT / f"server-{name}-{arm}.log")
        try:
            with httpx.Client(timeout=600) as c:
                for ids in enc["en"][:2]:  # warm-up
                    c.post(
                        url, json={"model": "m", "prompt": ids, "max_tokens": 32, "temperature": 0}
                    )
                for lang in LANGS:
                    d0, a0 = metrics() if spec else (0.0, 0.0)
                    tps, reps, ntoks = [], [], []
                    for item, ids in zip(prompts[lang], enc[lang], strict=True):
                        body = {
                            "model": "m",
                            "prompt": ids,
                            "max_tokens": max_tokens,
                            "temperature": 0,
                            "seed": 1234,
                        }
                        if fixed_len:  # block end-of-answer until max_tokens: equal lengths
                            body["min_tokens"] = max_tokens
                        t = time.perf_counter()
                        r = c.post(url, json=body).json()
                        dt_s = time.perf_counter() - t
                        k = r["usage"]["completion_tokens"]
                        out_ids = tok.encode(r["choices"][0]["text"], add_special_tokens=False)
                        tps.append(k / dt_s)
                        reps.append(rep4(out_ids))
                        ntoks.append(k)
                        rec = {
                            "model": name + ("+fixedlen" if fixed_len else ""),
                            "repo": repo,
                            "arm": arm,
                            "lang": lang,
                            "item": f"{item['task']}:{item['item_id']}",
                            "prompt_tokens": len(ids),
                            "output_tokens": k,
                            "latency_s": dt_s,
                            "rep4": rep4(out_ids),
                            "max_tokens": max_tokens,
                            "fixed_len": fixed_len,
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        with open(OUT / "speed.jsonl", "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(rec) + "\n")
                    d1, a1 = metrics() if spec else (0.0, 0.0)
                    tau = (a1 - a0 + d1 - d0) / (d1 - d0) if d1 > d0 else None
                    summary[(arm, lang)] = (
                        sum(tps) / len(tps),
                        tau,
                        sum(reps) / len(reps),
                        sum(ntoks) / len(ntoks),
                    )
        finally:
            stop(p)
    tag = " FIXED LENGTH (min_tokens = max_tokens)" if fixed_len else ""
    print(f"\n{name} ({repo}); n={len(keep)}/language, max_tokens={max_tokens}{tag}")
    print("lang  speedup   tau  tok/s(A0)  rep4(A0)  out_tok(A0)")
    for lang in LANGS:
        b, s = summary[("A0", lang)], summary[("ngram8", lang)]
        tau_s = f"{s[1]:.2f}" if s[1] else "-"
        print(f"{lang:5} {s[0] / b[0]:6.2f}x {tau_s:>5} {b[0]:9.1f} {b[2]:9.2f} {b[3]:11.0f}")
    sys.stdout.flush()


def speed_cmd(
    names: list[str], n: int, max_tokens: int, deadline: float | None, fixed_len: bool = False
) -> None:
    if not power_ok():
        sys.exit("GPU power limit below 140 W: switch to Turbo")
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = pick(bench(), 2 * n)  # spare candidates for models with short contexts
    for name in names:
        need = (11 if name in BIG else 7) * 60
        while not ready(MODELS[name][0]) and not (deadline and time.time() + need > deadline):
            time.sleep(10)
        if deadline and time.time() + need > deadline:
            print(f"\nSKIP {name}: not enough time left before the cutoff", flush=True)
            continue
        print(f"\n--- {time.strftime('%H:%M:%S')} start {name}", flush=True)
        try:
            one_model(name, prompts, max_tokens, n, fixed_len)
        except Exception as e:
            print(f"\nFAILED {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            subprocess.run(["pkill", "-f", "vllm serve"], check=False)
            time.sleep(10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tokens", "speed"])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--fixed-len", action="store_true", help="force every answer to max_tokens")
    ap.add_argument("--until", default=None, help="HH:MM: no model starts unless it can finish")
    a = ap.parse_args()
    deadline = None
    if a.until:
        now = dt.datetime.now()
        h, m = map(int, a.until.split(":"))
        d = now.replace(hour=h, minute=m, second=0, microsecond=0)
        deadline = (d if d > now else d + dt.timedelta(days=1)).timestamp()
    if a.cmd == "tokens":
        tokens_cmd()
    else:
        speed_cmd(a.models or list(MODELS), a.n, a.max_tokens, deadline, a.fixed_len)


if __name__ == "__main__":
    main()
