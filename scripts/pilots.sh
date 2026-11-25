#!/usr/bin/env bash
# Pilot sweeps (pilot split only), sequential.
L=$HOME/.cache/indicspec/logs/pilots.log; : > "$L"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
for c in configs/pilot_vllm_4b.yaml configs/pilot_vllm_8b.yaml; do
  echo "=== $(date +%T) START $c" >> "$L"
  bash scripts/sweep.sh "$c" >> "$L" 2>&1
  echo "=== exit $? $c $(date +%T)" >> "$L"
done
echo "=== PILOTS DONE $(date +%T)" >> "$L"
