#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/indicspec/venv-cpu HF_HUB_OFFLINE=1
uv run python -m indicspec.analysis.breakdown --run "${1:-confirm_vllm_4b}"
