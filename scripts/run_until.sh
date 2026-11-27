#!/usr/bin/env bash
# Run sweeps in order and stop cleanly after a wall-clock deadline (WSL).
# Usage: bash scripts/run_until.sh HH:MM configs/a.yaml [configs/b.yaml ...]
# Sweeps are resumable by cell, so re-running the same command continues where it stopped.
# Graceful: after the deadline no new sweep starts, and the running one is stopped as soon as its
# current cell finishes (a new line in any results/raw/*/*/cells.jsonl), so no measured cell is lost.
# Safety cap: if no cell finishes within MAX_OVERRUN_MIN (default 20) of the deadline, stop anyway.
# The deadline is checked against the clock every 20 s instead of one long `sleep`, which drifted by
# 10 minutes overnight under WSL2 (the VM's monotonic clock pauses when Windows throttles it).
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
DEADLINE="$1"; shift
L=$HOME/.cache/indicspec/logs/run_until.log
log() { echo "=== $(date '+%F %T') $*" >> "$L"; }
count() { cat results/raw/*/*/cells.jsonl 2>/dev/null | wc -l; }
target=$(date -d "$DEADLINE" +%s); [ "$target" -le "$(date +%s)" ] && target=$(( target + 86400 ))
cap=$(( target + 60 * ${MAX_OVERRUN_MIN:-20} ))
log "start; deadline $(date -d "@$target" '+%F %T') (graceful, cap +${MAX_OVERRUN_MIN:-20} min); configs: $*"
(
  while [ "$(date +%s)" -lt "$target" ]; do sleep 20; done
  n0=$(count); log "DEADLINE reached with $n0 cells; letting the running cell finish"
  while [ "$(count)" -le "$n0" ] && [ "$(date +%s)" -lt "$cap" ]; do
    pgrep -f "indicspec.harness.sweep" > /dev/null || break
    sleep 5
  done
  log "stopping ($(count) cells)"; bash scripts/stop_sweeps.sh >> "$L" 2>&1
) &
watchdog=$!
for c in "$@"; do
  [ "$(date +%s)" -ge "$target" ] && break
  log "START $c"; bash scripts/sweep.sh "$c" >> "$L" 2>&1; log "exit $? $c"
done
# After a deadline stop the loop ends while the watchdog is still inside stop_sweeps.sh; killing it
# then left the engine server running (2026-11-27). So wait for it; otherwise cancel it.
if [ "$(date +%s)" -ge "$target" ]; then wait "$watchdog"; else kill "$watchdog" 2>/dev/null; fi
log "DONE"
