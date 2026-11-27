#!/usr/bin/env bash
# Rebuild every results table that comes from raw logs, then check it matches the committed copy.
# Usage (WSL): bash scripts/reproduce.sh            # CPU only; needs the Qwen3 tokenizer in HF cache
# Prints one line per table: "same" or "CHANGED" (with a diff stat). Exit code 1 if anything changed.
# Not covered (need the benchmark data or drafter checkpoints, not just raw logs): token_tax,
# fragmentation_ref, coverage_ref, vocab_classes, oracle_tau_bound, graft_*. Figures are separate
# (scripts/figures.sh).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/indicspec/venv-cpu HF_HUB_OFFLINE=1
uv sync -q --extra analysis --extra data
py() { echo "+ python -m $*"; uv run python -m "$@" > /dev/null || { echo "FAILED: $*"; exit 2; }; }

for r in pilot_vllm_4b pilot_vllm_8b; do [ -d "results/raw/$r" ] && py indicspec.analysis.pilot --run "$r"; done
for r in tune_vllm_4b tune_vllm_8b; do py indicspec.analysis.tuning --run "$r"; done
for r in confirm_vllm_4b confirm_vllm_8b; do
  # a run is only analysed once complete (288/288 cells for 8B, 324 for 4B); partial runs are skipped
  n=$(cat results/raw/$r/*/cells.jsonl 2>/dev/null | wc -l)
  want=$( [ "$r" = confirm_vllm_4b ] && echo 324 || echo 288 )
  if [ "$n" -ge "$want" ]; then
    py indicspec.analysis.confirm --run "$r"
    py indicspec.analysis.breakdown --run "$r"
  else
    echo "skip $r ($n/$want cells)"
  fi
done
py indicspec.analysis.xengine

echo "--- comparison with the committed tables ---"
bad=0
for f in results/tables/*; do
  if ! git ls-files --error-unmatch "$f" > /dev/null 2>&1; then echo "NEW (untracked)  $f"; continue; fi
  if git diff --quiet -- "$f"; then echo "same      $f"; else echo "CHANGED   $f"; git diff --stat -- "$f" | head -1; bad=1; fi
done
exit $bad
