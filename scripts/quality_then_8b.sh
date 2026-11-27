#!/usr/bin/env bash
# Answer-quality check for the kept quick-test models, then the 8B confirmatory sweep (WSL).
# Usage: bash scripts/quality_then_8b.sh 11:00
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
END="$1"
L=$HOME/.cache/indicspec/logs/quick_models.log
PY=$HOME/.cache/indicspec/venv-vllm/bin/python
echo "=== $(date '+%F %T') quality check" >> "$L"
HF_HUB_OFFLINE=1 "$PY" scripts/quick_quality.py --n 20 qwen3-4b qwen3-0.6b gemma-4-e4b sarvam-1 \
  navarasa-7b tamil-llama-7b openhathi-7b lumma-0.6b >> "$L" 2>&1
bash scripts/stop_sweeps.sh >> "$L" 2>&1
echo "=== $(date '+%F %T') starting 8B confirmatory run until $END" >> "$L"
bash scripts/run_until.sh "$END" configs/confirm_vllm_8b.yaml
echo "=== $(date '+%F %T') quality + 8B chain done" >> "$L"
