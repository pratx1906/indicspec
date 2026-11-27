#!/usr/bin/env bash
# FrontiersMind quick test (WSL): wait until the GPU is free, check vLLM correctness for Lumma,
# then the standard quick speed test for Qwen3-0.6B (reference), Lumma-0.6B and Nandi-Mini-150M.
set -uo pipefail
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
L=$HOME/.cache/indicspec/logs/quick_models.log
PY=$HOME/.cache/indicspec/venv-vllm/bin/python
echo "=== $(date '+%F %T') FrontiersMind job: waiting for a free GPU" >> "$L"
while pgrep -f "indicspec.harness.sweep|vllm serve" > /dev/null; do sleep 10; done
sleep 5
echo "=== $(date '+%F %T') vLLM correctness check: lumma-0.6b" >> "$L"
"$PY" scripts/quick_models_check.py lumma-0.6b >> "$L" 2>&1
pkill -f "vllm serve"; sleep 5
echo "=== $(date '+%F %T') speed test: qwen3-0.6b lumma-0.6b nandi-mini-150m" >> "$L"
"$PY" scripts/quick_models.py speed qwen3-0.6b lumma-0.6b nandi-mini-150m >> "$L" 2>&1
echo "=== $(date '+%F %T') FrontiersMind job done" >> "$L"
