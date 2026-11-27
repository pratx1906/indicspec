#!/usr/bin/env bash
# Run sweeps in order and stop everything cleanly at a wall-clock deadline (WSL).
# Usage: bash scripts/run_until.sh HH:MM configs/a.yaml [configs/b.yaml ...]
# Sweeps are resumable by cell, so re-running the same command continues where it stopped.
# The deadline is checked against the clock every 20 s instead of one long `sleep`, which drifted by
# 10 minutes overnight under WSL2 (the VM's monotonic clock pauses when Windows throttles it).
export PATH="$HOME/.local/bin:$PATH"
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
DEADLINE="$1"; shift
L=$HOME/.cache/indicspec/logs/run_until.log
log() { echo "=== $(date '+%F %T') $*" >> "$L"; }
target=$(date -d "$DEADLINE" +%s); [ "$target" -le "$(date +%s)" ] && target=$(( target + 86400 ))
log "start; deadline $(date -d "@$target" '+%F %T'); configs: $*"
(
  while [ "$(date +%s)" -lt "$target" ]; do sleep 20; done
  log "DEADLINE reached: stopping"; bash scripts/stop_sweeps.sh >> "$L" 2>&1
) &
watchdog=$!
for c in "$@"; do
  [ "$(date +%s)" -ge "$target" ] && break
  log "START $c"; bash scripts/sweep.sh "$c" >> "$L" 2>&1; log "exit $? $c"
done
# At the deadline the sweep dies first and the loop ends while the watchdog is still inside
# stop_sweeps.sh; killing it then left the engine server running (2026-11-27). So: wait for it.
if [ "$(date +%s)" -ge "$target" ]; then wait "$watchdog"; else kill "$watchdog" 2>/dev/null; fi
log "DONE"
