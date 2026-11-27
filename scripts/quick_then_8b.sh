#!/usr/bin/env bash
# One unattended chain (WSL): cross-model quick test until a cutoff, token test, then the 8B
# confirmatory sweep until a graceful deadline.
# Usage: bash scripts/quick_then_8b.sh 23:40 10:00
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
CUT="$1"; END="$2"
L=$HOME/.cache/indicspec/logs/quick_models.log
PY=$HOME/.cache/indicspec/venv-vllm/bin/python
echo "=== $(date '+%F %T') chain start: quick test until $CUT, then 8B until $END" >> "$L"
bash scripts/quick_models_download.sh &
"$PY" scripts/quick_models.py speed --until "$CUT" >> "$L" 2>&1
echo "=== $(date '+%F %T') speed test done; token test" >> "$L"
HF_HUB_OFFLINE=1 "$PY" scripts/quick_models.py tokens >> "$L" 2>&1
bash scripts/stop_sweeps.sh >> "$L" 2>&1
echo "=== $(date '+%F %T') starting 8B confirmatory run until $END" >> "$L"
bash scripts/run_until.sh "$END" configs/confirm_vllm_8b.yaml
echo "=== $(date '+%F %T') chain done" >> "$L"
