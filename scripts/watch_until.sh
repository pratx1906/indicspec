#!/usr/bin/env bash
# Quiet watcher: exits (one line) on the first failure signal, or at the deadline (HH:MM, default 05:00).
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
DEADLINE="${1:-05:00}"; L=$HOME/.cache/indicspec/logs
while true; do
  f=$(grep -hE "FAILED|Traceback" "$L/overnight.log" "$L/tuning.log" 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "ALERT $(date +%T): $f"; exit 1; }
  for b in results/raw/tune_*/block_failures.jsonl results/raw/confirm_*/block_failures.jsonl; do
    [ -s "$b" ] && { echo "ALERT $(date +%T): block failure in $b"; exit 1; }
  done
  grep -q "OVERNIGHT DONE" "$L/overnight.log" 2>/dev/null && { echo "DONE $(date +%T)"; exit 0; }
  [ "$(date +%H:%M)" \> "$DEADLINE" ] && [ "$(date +%H)" -lt 12 ] && {
    echo "OK at $(date +%T): $(tail -1 "$L/overnight.log" 2>/dev/null)"; exit 0; }
  sleep 300
done
