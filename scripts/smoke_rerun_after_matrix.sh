#!/usr/bin/env bash
# Wait for the smoke matrix to finish, then re-run arms given as args (engine:arm:model ...).
L=$HOME/.cache/indicspec/logs/smoke_matrix.log
WAIT_FOR="${WAIT_FOR:-MATRIX DONE}"
until grep -q "$WAIT_FOR" "$L"; do sleep 20; done
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
for spec in "$@"; do
  IFS=: read -r e a m <<< "$spec"
  echo "=== $(date +%T) RERUN $e $a $m" >> "$L"
  bash scripts/smoke.sh "$e" "$a" "$m" >> "$L" 2>&1
  echo "=== exit $? RERUN $e $a $m" >> "$L"
  sleep 5
done
echo "=== ${DONE_TAG:-RERUNS DONE} $(date +%T)" >> "$L"
