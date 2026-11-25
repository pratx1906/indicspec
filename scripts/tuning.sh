#!/usr/bin/env bash
# Draft-length tuning sweeps (tuning split), sequential.
L=$HOME/.cache/indicspec/logs/tuning.log; : > "$L"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
for c in configs/tune_vllm_4b.yaml configs/tune_vllm_8b.yaml; do
  echo "=== $(date +%T) START $c" >> "$L"
  bash scripts/sweep.sh "$c" >> "$L" 2>&1
  echo "=== exit $? $c $(date +%T)" >> "$L"
done
echo "=== TUNING DONE $(date +%T)" >> "$L"
