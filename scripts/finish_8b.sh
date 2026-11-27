#!/usr/bin/env bash
# One command after the Qwen3-8B confirmatory sweep finishes (WSL, CPU only, ~10-20 min).
# Usage: bash scripts/finish_8b.sh
#   1. refuses to run unless confirm_vllm_8b has all 288 cells and nothing is sweeping
#   2. confirmatory analysis over BOTH models (H1 needs 8B; G1/G2 over all pairs)
#   3. exploratory breakdowns for 8B (by task, energy, token class)
#   4. figures, then a plain-language digest -> results/tables/confirm_vllm_4b+confirm_vllm_8b_digest.md
# It does not edit CLAIMS/README/DECISIONS and does not commit: those stay a reviewed, manual step.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/indicspec/venv-cpu HF_HUB_OFFLINE=1
L=$HOME/.cache/indicspec/logs/finish_8b.log
exec > >(tee -a "$L") 2>&1
echo "=== $(date '+%F %T') finish_8b"

n=$(cat results/raw/confirm_vllm_8b/*/cells.jsonl 2>/dev/null | wc -l)
[ "$n" -ge 288 ] || { echo "confirm_vllm_8b has $n/288 cells: not finished, nothing done"; exit 1; }
if pgrep -f "indicspec.harness.sweep" > /dev/null; then echo "a sweep is still running: stop"; exit 1; fi
[ -s results/raw/confirm_vllm_8b/block_failures.jsonl ] && { echo "WARNING: block failures:"; cut -c1-300 results/raw/confirm_vllm_8b/block_failures.jsonl; }

uv sync -q --extra analysis --extra data
NAME="confirm_vllm_4b+confirm_vllm_8b"
echo "--- confirmatory analysis ($NAME)"
uv run python -m indicspec.analysis.confirm --run confirm_vllm_4b --run confirm_vllm_8b --name "$NAME"
echo "--- breakdowns (8B)"
uv run python -m indicspec.analysis.breakdown --run confirm_vllm_8b
echo "--- figures"
uv run python -m indicspec.analysis.figures > /dev/null
echo "--- digest"
uv run python -m indicspec.analysis.digest --name "$NAME" --out "results/tables/${NAME}_digest.md"
echo "=== $(date '+%F %T') done; review the digest, then update CLAIMS/README/DECISIONS by hand"
