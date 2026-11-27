#!/usr/bin/env bash
# Exploratory sampled + long-answer sweeps on Qwen3-4B (D-026), queued behind the 8B confirmatory run
# (WSL). Waits until scripts/run_then_finish_8b.sh has exited, then runs both configs with a graceful
# deadline, then the pilot-style analysis (speed-ups and tau with bootstrap CIs) for each.
# Usage: bash scripts/explore_after_8b.sh HH:MM
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
L=$HOME/.cache/indicspec/logs/run_until.log
echo "=== $(date '+%F %T') explore_after_8b: waiting for the 8B run to finish" >> "$L"
while pgrep -f "run_then_finish_8b.sh" > /dev/null || pgrep -f "indicspec.harness.sweep" > /dev/null; do
  sleep 30
done
bash scripts/run_until.sh "$1" configs/explore_sampled_vllm_4b.yaml configs/explore_long_vllm_4b.yaml
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/indicspec/venv-cpu HF_HUB_OFFLINE=1
for r in explore_sampled_vllm_4b explore_long_vllm_4b; do
  [ -d "results/raw/$r" ] && uv run python -m indicspec.analysis.pilot --run "results/raw/$r" >> "$L" 2>&1
done
echo "=== $(date '+%F %T') explore_after_8b: done" >> "$L"
