#!/usr/bin/env bash
# Sequential smoke matrix (one server owns the GPU at a time). WSL only.
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
LOG=$HOME/.cache/indicspec/logs/smoke_matrix.log
run() { echo "=== $(date +%T) $*" >> "$LOG"; bash scripts/smoke.sh "$@" >> "$LOG" 2>&1; echo "=== exit $? $*" >> "$LOG"; sleep 5; }
for e in vllm sglang; do
  for a in A0 A1 A5; do run "$e" "$a" qwen3-4b; done
  for a in A0 A3 A4 A5; do run "$e" "$a" qwen3-8b; done
done
echo "=== MATRIX DONE $(date +%T)" >> "$LOG"
