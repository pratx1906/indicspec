#!/usr/bin/env bash
# Confirmatory chain with a hard stop: confirm_vllm_4b -> 4B analysis (CPU, background) -> confirm_vllm_8b,
# everything stopped cleanly at DEADLINE (HH:MM, default 11:00). Sweeps are resumable by cell.
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/indicspec/venv-cpu HF_HUB_OFFLINE=1
DEADLINE="${1:-11:00}"
L=$HOME/.cache/indicspec/logs/confirm_chain.log; : > "$L"
log() { echo "=== $(date '+%F %T') $*" >> "$L"; }
secs=$(( $(date -d "$DEADLINE" +%s) - $(date +%s) )); [ "$secs" -le 0 ] && secs=$(( secs + 86400 ))
( sleep "$secs"; echo "=== $(date '+%F %T') DEADLINE $DEADLINE: stopping" >> "$L"; bash scripts/stop_sweeps.sh >> "$L" 2>&1 ) &
log "start (deadline $DEADLINE, in ${secs}s)"
log "START confirm_vllm_4b"; bash scripts/sweep.sh configs/confirm_vllm_4b.yaml >> "$L" 2>&1; log "exit $? confirm_vllm_4b"
( uv run python -m indicspec.analysis.confirm --run confirm_vllm_4b >> "$HOME/.cache/indicspec/logs/confirm_analysis_4b.log" 2>&1; log "4B analysis exit $?" ) &
log "START confirm_vllm_8b"; bash scripts/sweep.sh configs/confirm_vllm_8b.yaml >> "$L" 2>&1; log "exit $? confirm_vllm_8b"
wait
log "CHAIN DONE"
