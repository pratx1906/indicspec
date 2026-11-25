cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,clocks_event_reasons.active,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown,clocks_event_reasons.gpu_idle --format=csv
f=$(ls -t results/raw/pilot_vllm_4b/*/telemetry-*.csv | head -1); echo "telemetry: $f"
awk -F', ' 'NR>1 && $7+0>50 {n++; c[n]=$2; p+=$4; t=($5>t?$5:t)} END {asort(c); print "busy samples", n, "SM clock p5/median/p95:", c[int(n*0.05)+1], c[int(n*0.5)], c[int(n*0.95)], "mean power", p/n, "max temp", t}' "$f"
awk -F', ' 'NR>1 && $7+0>50 {print $8}' "$f" | sort | uniq -c | sort -rn | head -5
