#!/usr/bin/env bash
# Resume the Qwen3-8B confirmatory sweep with a graceful deadline, then (only if all 288 cells exist)
# run the CPU-only finish step. finish_8b.sh refuses on its own if the sweep is incomplete.
# Usage (WSL): bash scripts/run_then_finish_8b.sh HH:MM
set -uo pipefail
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
bash scripts/run_until.sh "$1" configs/confirm_vllm_8b.yaml
bash scripts/finish_8b.sh
