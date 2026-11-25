#!/usr/bin/env bash
# After tuning: choose draft settings (frozen rule) -> correctness gate -> confirmatory sweeps.
# Stops at the first failing gate; never runs confirmatory sweeps on an arm that failed equality.
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/indicspec/venv-cpu HF_HUB_OFFLINE=1
L=$HOME/.cache/indicspec/logs/overnight.log; : > "$L"
log() { echo "=== $(date +%T) $*" >> "$L"; }
until grep -q "TUNING DONE" "$HOME/.cache/indicspec/logs/tuning.log" 2>/dev/null; do sleep 30; done
log "tuning done; choosing settings"
for r in tune_vllm_4b tune_vllm_8b; do
  uv run python -m indicspec.analysis.tuning --run "$r" --write-config >> "$L" 2>&1 || { log "TUNING ANALYSIS FAILED $r"; exit 1; }
done
log "equality gate 4b"
uv run python -m indicspec.harness.equality_gate --model qwen3-4b --arms A0 A1 A5 >> "$L" 2>&1 || { log "GATE FAILED qwen3-4b"; exit 1; }
log "equality gate 8b"
uv run python -m indicspec.harness.equality_gate --model qwen3-8b --arms A0 A3 A4 A5 >> "$L" 2>&1 || { log "GATE FAILED qwen3-8b"; exit 1; }
for c in configs/confirm_vllm_4b.yaml configs/confirm_vllm_8b.yaml; do
  log "START $c"
  bash scripts/sweep.sh "$c" >> "$L" 2>&1
  log "exit $? $c"
done
log "OVERNIGHT DONE"
