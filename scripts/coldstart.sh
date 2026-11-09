#!/usr/bin/env bash
# Measure cold start time: delete pod, measure time until /readyz returns 200.
# Runs 3 times and reports median.

set -euo pipefail

NAMESPACE="app"
LABEL="app=command-parser"
HOST="${HOST:-http://localhost:8080}"
OUTFILE="results/coldstart_times.txt"

mkdir -p results
echo "Cold start measurements" > "$OUTFILE"

times=()

for i in 1 2 3; do
    echo "=== Cold start run $i/3 ==="

    # Scale to 1 replica first
    kubectl -n "$NAMESPACE" scale deploy/command-parser --replicas=1
    kubectl -n "$NAMESPACE" rollout status deploy/command-parser --timeout=5m

    # Delete the pod
    POD=$(kubectl -n "$NAMESPACE" get pods -l "$LABEL" -o jsonpath='{.items[0].metadata.name}')
    echo "Deleting pod: $POD"
    START=$(date +%s)
    kubectl -n "$NAMESPACE" delete pod "$POD"

    # Wait for readiness
    echo -n "Waiting for readyz..."
    while true; do
        CODE=$(curl -s -o /dev/null -w '%{http_code}' "$HOST/readyz" 2>/dev/null || echo "000")
        if [ "$CODE" = "200" ]; then
            break
        fi
        echo -n "."
        sleep 1
    done
    END=$(date +%s)

    ELAPSED=$(( END - START ))  # seconds
    echo " ready in ${ELAPSED}s"
    echo "Run $i: ${ELAPSED}s" >> "$OUTFILE"
    times+=("$ELAPSED")

    sleep 5
done

# Calculate median
IFS=$'\n' sorted=($(sort -n <<<"${times[*]}")); unset IFS
MEDIAN=${sorted[1]}
echo "Median cold start: ${MEDIAN}s" | tee -a "$OUTFILE"

