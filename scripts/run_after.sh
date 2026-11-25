#!/usr/bin/env bash
# Usage (WSL): bash scripts/run_after.sh <log-file> <marker> <command...>
# Waits until <marker> appears in <log-file>, then runs the command (GPU jobs never overlap).
LOG="$1"; MARK="$2"; shift 2
until grep -q "$MARK" "$LOG" 2>/dev/null; do sleep 20; done
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
"$@"
