#!/usr/bin/env bash
# Rest of the quick-test chain (WSL): remaining speed tests until a cutoff, Sarvam-1 at fixed
# length, the token test, then the 8B confirmatory sweep until a graceful deadline.
# Usage: bash scripts/quick_tail_8b.sh 23:40 10:00 MODEL [MODEL ...]
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
CUT="$1"; END="$2"; shift 2
L=$HOME/.cache/indicspec/logs/quick_models.log
PY=$HOME/.cache/indicspec/venv-vllm/bin/python
echo "=== $(date '+%F %T') chain (resumed): $* until $CUT, then 8B until $END" >> "$L"
bash scripts/quick_models_download.sh &
"$PY" scripts/quick_models.py speed "$@" --until "$CUT" >> "$L" 2>&1
echo "=== $(date '+%F %T') Sarvam-1 at fixed length" >> "$L"
"$PY" scripts/quick_models.py speed sarvam-1 --fixed-len --until 23:59 >> "$L" 2>&1
echo "=== $(date '+%F %T') token test" >> "$L"
HF_HUB_OFFLINE=1 "$PY" scripts/quick_models.py tokens >> "$L" 2>&1
bash scripts/stop_sweeps.sh >> "$L" 2>&1
echo "=== $(date '+%F %T') starting 8B confirmatory run until $END" >> "$L"
bash scripts/run_until.sh "$END" configs/confirm_vllm_8b.yaml
echo "=== $(date '+%F %T') chain done" >> "$L"
