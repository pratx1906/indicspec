#!/usr/bin/env bash
# Download the models for the exploratory cross-model quick test, in priority order (WSL; network
# only). Tokenizer/config files for all models first (for the CPU token test), then weights one model
# at a time; a marker file per finished repo tells scripts/quick_models.py it can start that model.
set -uo pipefail
PY=$HOME/.cache/indicspec/venv-vllm/bin/python
L=$HOME/.cache/indicspec/logs/quick_models_download.log
M=$HOME/.cache/indicspec/quick_models_ready; mkdir -p "$M"
REPOS=(Qwen/Qwen3-4B google/gemma-4-E4B-it sarvamai/sarvam-1 Qwen/Qwen2.5-1.5B-Instruct
  eulogik/Bharat-Tiny-LLM-v2 krutrim-ai-labs/Krutrim-1-instruct abhinand/tamil-llama-7b-instruct-v0.2
  Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0 abhinand/malayalam-llama-7b-instruct-v0.1
  sarvamai/OpenHathi-7B-Hi-v0.1-Base google/gemma-4-E2B-it soketlabs/pragna-1b LingoIITGN/Ganga-2-1B
  nickmalhotra/ProjectIndus meta-llama/Llama-2-7b-hf)
dl() {  # repo, patterns...
  "$PY" - "$@" >> "$L" 2>&1 <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, pats = sys.argv[1], sys.argv[2:]
print("ok", repo, snapshot_download(repo, allow_patterns=pats))
EOF
}
echo "=== $(date '+%T') tokenizers" >> "$L"
for r in "${REPOS[@]:1}"; do dl "$r" "*.json" "*.model" "*.txt" "*.tiktoken" "*.jinja" || echo "FAIL tok $r" >> "$L"; done
touch "$M/Qwen--Qwen3-4B"   # already cached at the pinned revision used by the study
for r in "${REPOS[@]}"; do
  [ "$r" = Qwen/Qwen3-4B ] && continue
  echo "=== $(date '+%T') weights $r" >> "$L"
  # safetensors when available, else pytorch .bin (older repos)
  if dl "$r" "*.json" "*.model" "*.txt" "*.jinja" "*.safetensors" && \
     ls "$HOME/.cache/huggingface/hub/models--${r//\//--}/snapshots/"*/*.safetensors > /dev/null 2>&1; then
    touch "$M/${r//\//--}"
  elif dl "$r" "*.json" "*.model" "*.txt" "*.jinja" "*.bin"; then
    touch "$M/${r//\//--}"
  else
    echo "FAIL weights $r" >> "$L"
  fi
done
echo "=== $(date '+%T') ALL DONE" >> "$L"
