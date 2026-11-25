#!/usr/bin/env bash
# Progress of running sweeps (WSL): cells done per block, failures, GPU state.
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
date +%T
cat $HOME/.cache/indicspec/logs/pilots.log $HOME/.cache/indicspec/logs/tuning.log $HOME/.cache/indicspec/logs/overnight.log 2>/dev/null | grep -E "^===" 2>/dev/null
for run in results/raw/tune_* results/raw/confirm_*; do
  [ -d "$run" ] || continue
  for f in "$run"/*/cells.jsonl; do
    [ -f "$f" ] && echo "$(basename "$run")/$(basename "$(dirname "$f")"): $(wc -l < "$f") cells"
  done
  [ -s "$run/block_failures.jsonl" ] && { echo "FAILURES in $run:"; cut -c1-300 "$run/block_failures.jsonl"; }
done
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm --format=csv,noheader
