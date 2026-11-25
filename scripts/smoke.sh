#!/usr/bin/env bash
# Usage (WSL): bash scripts/smoke.sh <engine> <arm> <model> [extra smoke args]
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"   # uv; non-login shells lack it
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/indicspec/venv-cpu
export HF_HUB_OFFLINE=1
uv run python -m indicspec.harness.smoke --engine "$1" --arm "$2" --model "$3" "${@:4}"
