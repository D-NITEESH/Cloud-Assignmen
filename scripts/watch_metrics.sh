#!/usr/bin/env bash
# Sample kubectl top pods at intervals for the specified duration.
#
# Usage:
#   bash scripts/watch_metrics.sh 150   # sample for 150 seconds

set -euo pipefail

DURATION="${1:-120}"
INTERVAL=5
OUTFILE="results/metrics_samples.csv"

mkdir -p results

echo "timestamp,name,cpu,memory" > "$OUTFILE"
echo "Sampling pod metrics for ${DURATION}s (every ${INTERVAL}s) ..."

END=$(($(date +%s) + DURATION))

while [ "$(date +%s)" -lt "$END" ]; do
    TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    kubectl -n app top pods --no-headers -l app=command-parser 2>/dev/null | while read -r NAME CPU MEM; do
        echo "${TS},${NAME},${CPU},${MEM}" >> "$OUTFILE"
    done
    sleep "$INTERVAL"
done

echo "Metrics saved to $OUTFILE"

