#!/usr/bin/env bash
# Graceful stop (WSL): after a wall-clock deadline, wait for the running cell to finish (a new line in
# any cells.jsonl of the run), then stop the sweep and its engine server. No measured cell is lost;
# only the next cell's first seconds are discarded.
# Usage: bash scripts/stop_after_cell.sh HH:MM results/raw/<run>
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
DEADLINE="$1"; RUN="$2"
L=$HOME/.cache/indicspec/logs/run_until.log
log() { echo "=== $(date '+%F %T') $*" >> "$L"; }
target=$(date -d "$DEADLINE" +%s); [ "$target" -le "$(date +%s)" ] && target=$(( target + 86400 ))
count() { cat "$RUN"/*/cells.jsonl 2>/dev/null | wc -l; }
log "graceful stop armed: after $(date -d "@$target" '+%F %T'), at the next completed cell of $RUN"
while [ "$(date +%s)" -lt "$target" ]; do sleep 20; done
n0=$(count); log "deadline reached with $n0 cells; waiting for the running cell to finish"
while [ "$(count)" -le "$n0" ]; do
  pgrep -f "indicspec.harness.sweep" > /dev/null || { log "sweep already ended"; break; }
  sleep 5
done
log "cell finished ($(count) cells): stopping"
bash scripts/stop_sweeps.sh >> "$L" 2>&1
