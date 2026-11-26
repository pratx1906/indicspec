#!/usr/bin/env bash
# One-off: run the 8B greedy-equality gate (diagnostic after the 4B gate stopped the overnight chain).
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/indicspec/venv-cpu HF_HUB_OFFLINE=1
L=$HOME/.cache/indicspec/logs/gate_8b.log
echo "=== $(date +%T) start" > "$L"
uv run python -m indicspec.harness.equality_gate --model qwen3-8b --arms A0 A3 A4 A5 >> "$L" 2>&1
echo "=== $(date +%T) exit $?" >> "$L"
