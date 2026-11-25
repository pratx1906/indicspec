#!/usr/bin/env bash
# Regenerate README/report figures from results/tables (WSL/Linux).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/indicspec/venv-cpu
uv sync -q --extra analysis --extra data
uv run python -m indicspec.analysis.figures
