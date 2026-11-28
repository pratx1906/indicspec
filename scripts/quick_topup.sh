#!/usr/bin/env bash
# Convergence top-up of the cross-model quick test (D-028): the four reliable models, 40 items per
# language (was 10), same measures. New rows are appended to results/raw/quick_models/speed.jsonl
# with "n": 40 in their time window (the summary separates them by time). Usage (WSL): bash scripts/quick_topup.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
L=$HOME/.cache/indicspec/logs/quick_models.log
echo "=== $(date '+%F %T') convergence top-up start" >> "$L"
HF_HUB_OFFLINE=1 "$HOME/.cache/indicspec/venv-vllm/bin/python" scripts/quick_models.py speed \
  qwen3-4b navarasa-7b gemma-4-e4b sarvam-1 --n 40 --until 23:45 >> "$L" 2>&1
bash scripts/stop_sweeps.sh >> "$L" 2>&1
echo "=== $(date '+%F %T') convergence top-up done" >> "$L"
