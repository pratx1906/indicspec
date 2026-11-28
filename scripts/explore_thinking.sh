#!/usr/bin/env bash
# Sequential thinking-mode study (D-028), WSL. Waits for any running sweep to end, then runs one stage
# config with a graceful deadline. Usage: bash scripts/explore_thinking.sh HH:MM configs/<stage>.yaml
set -uo pipefail
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
while pgrep -f "indicspec.harness.sweep" > /dev/null || pgrep -f "stop_after_cell.sh" > /dev/null; do sleep 15; done
bash scripts/run_until.sh "$1" "$2"
