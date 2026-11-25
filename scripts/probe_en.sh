#!/usr/bin/env bash
# English configuration probes (WS1.6): pilot-split + hand-written prompts, 512 tokens.
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
L=$HOME/.cache/indicspec/logs/probe_en.log; : > "$L"
P=data/bench/probes_en.jsonl
run() { echo "=== $(date +%T) $*" >> "$L"; bash scripts/smoke.sh "$@" --prompts-file $P --max-tokens 512 >> "$L" 2>&1; echo "=== exit $? $*" >> "$L"; sleep 5; }
run vllm A5 qwen3-8b --tag probe-en-k15
run vllm A5 qwen3-8b --tag probe-en-k7 --spec-override '{"num_speculative_tokens": 7}'
run vllm A5 qwen3-4b --tag probe-en-k15
run vllm A3 qwen3-8b --tag probe-en-k3
run sglang A5 qwen3-8b --tag probe-en
run sglang A5 qwen3-4b --tag probe-en
echo "=== PROBES DONE $(date +%T)" >> "$L"
