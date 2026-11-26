#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/indicspec/venv-cpu HF_HUB_OFFLINE=1
L=$HOME/.cache/indicspec/logs/diag.log; : > "$L"
uv run python scripts/equality_diag.py --model qwen3-4b --arms A1 A5 >> "$L" 2>&1 || echo "FAILED 4b" >> "$L"
uv run python scripts/equality_diag.py --model qwen3-8b --arms A3 A4 A5 >> "$L" 2>&1 || echo "FAILED 8b" >> "$L"
tail -4 "$L"
