#!/usr/bin/env bash
# Usage (WSL): bash scripts/sweep.sh configs/<name>.yaml
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/indicspec/venv-cpu HF_HUB_OFFLINE=1
uv run python -m indicspec.harness.sweep --config "$1"
