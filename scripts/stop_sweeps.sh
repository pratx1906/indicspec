#!/usr/bin/env bash
# Stop queued/running pilot sweeps and their engine servers (WSL).
pkill -f "scripts/run_after.sh" ; pkill -f "scripts/pilots.sh" ; pkill -f "indicspec.harness.sweep"
sleep 3
pkill -f "vllm serve" ; pkill -f "sglang.launch_server" ; pkill -f "nvidia-smi --query-gpu"
sleep 5
pgrep -af "vllm|sglang|indicspec.harness" | grep -v pgrep || echo "all stopped"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
