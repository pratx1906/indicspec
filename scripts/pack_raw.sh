#!/usr/bin/env bash
# Losslessly gzip a FINISHED run's requests.jsonl files (keeps them under GitHub's 50 MB warning).
# Usage: bash scripts/pack_raw.sh results/raw/<run>
# Each file is compressed, the .gz is decompressed and its SHA-256 compared with the original, and
# only then is the plain file removed. Analyses read either form (src/indicspec/rawio.py).
# Never run this on a run that is still sweeping: the harness would start a new plain file.
set -euo pipefail
run="${1:?usage: pack_raw.sh results/raw/<run>}"
find "$run" -name requests.jsonl | while read -r f; do
  [ -e "$f.gz" ] && { echo "skip (gz exists): $f"; continue; }
  gzip -9 -n -k "$f"
  if [ "$(sha256sum < "$f")" = "$(gzip -dc "$f.gz" | sha256sum)" ]; then
    echo "$(du -h "$f" | cut -f1) -> $(du -h "$f.gz" | cut -f1)  $f"; rm "$f"
  else
    echo "MISMATCH, kept plain: $f" >&2; rm "$f.gz"; exit 1
  fi
done
