#!/usr/bin/env bash
# 20 s bf16 matmul load; samples power/clocks/throttle reasons every second (WSL).
PY=$HOME/.cache/indicspec/venv-vllm/bin/python
nvidia-smi --query-gpu=power.draw,enforced.power.limit,clocks.sm,temperature.gpu,utilization.gpu,clocks_event_reasons.active --format=csv,noheader -l 1 > /tmp/gpuload.csv &
SMI=$!
$PY - <<'PY'
import time, torch
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
torch.cuda.synchronize(); t0 = time.time(); n = 0
while time.time() - t0 < 20:
    for _ in range(10): c = a @ b
    torch.cuda.synchronize(); n += 10
dt = time.time() - t0
print(f"sustained bf16 matmul: {2*8192**3*n/dt/1e12:.1f} TFLOPS over {dt:.0f}s")
PY
kill $SMI
echo "power W, limit W, SM MHz, temp C, util, reasons (last 12 s of load):"
tail -n 14 /tmp/gpuload.csv | head -n 12
powercfg.exe /getactivescheme 2>/dev/null | tr -d '\r'
