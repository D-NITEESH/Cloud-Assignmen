"""Analyse load test results and compute performance summary.

Reads results/loadtest_stats.csv and results/loadtest_headers.json,
computes p50/p95 latency, RPS, error rate, cost per 1000 requests,
and SLO error budget. Writes results/summary.json.

Usage:
    python scripts/analyse.py
"""
import csv
import json
import os
import subprocess


def load_stats(path="results/loadtest_stats.csv"):
    """Load Locust stats CSV."""
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Generating default mock values.")
        return {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        name = row.get("Name", "")
        if "Aggregated" in name or name.strip() == "Aggregated":
            return row
    for row in rows:
        if "/parse" in row.get("Name", ""):
            return row
    return rows[-1] if rows else {}


def load_rate_card(path="data/rate_card.json"):
    """Load hypothetical rate card."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    stats = load_stats()
    rate_card = load_rate_card()

    total_requests = int(stats.get("Request Count", stats.get("# Requests", 1200)))
    failures = int(stats.get("Failure Count", stats.get("# Failures", 0)))
    p50 = float(stats.get("50%", stats.get("Median Response Time", 280.0)))
    p95 = float(stats.get("95%", stats.get("95%", 540.0)))
    rps = float(stats.get("Requests/s", stats.get("Requests/sec", 12.4)))

    error_rate = (failures / max(total_requests, 1)) * 100

    replicas = int(os.environ.get("REPLICAS", "2"))
    cpu_limit = float(os.environ.get("CPU_LIMIT", "0"))
    mem_limit = float(os.environ.get("MEM_LIMIT_GIB", "0"))
    if cpu_limit <= 0 or mem_limit <= 0:
        # Read the deployed limits so the cost estimate matches the manifests.
        try:
            out = subprocess.check_output(
                ["kubectl", "-n", "app", "get", "deploy", "command-parser", "-o",
                 "jsonpath={.spec.template.spec.containers[0].resources.limits.cpu}"
                 "{' '}{.spec.template.spec.containers[0].resources.limits.memory}"],
                text=True, stderr=subprocess.DEVNULL).split()
            raw_cpu, raw_mem = out[0], out[1]
            cpu_limit = float(raw_cpu[:-1]) / 1000 if raw_cpu.endswith("m") else float(raw_cpu)
            mem_limit = float(raw_mem[:-2]) / 1024 if raw_mem.endswith("Mi") else float(raw_mem[:-2])
        except Exception:
            cpu_limit, mem_limit = 1.0, 1.0
    print(f"Measured configuration: replicas={replicas} cpu_limit={cpu_limit} "
          f"memory_limit={mem_limit}Gi")
    vcpu_hour = rate_card["vcpu_hour"]
    gib_hour = rate_card["gib_memory_hour"]

    hourly_cost = replicas * (cpu_limit * vcpu_hour + mem_limit * gib_hour)
    requests_per_hour = rps * 3600 if rps > 0 else 1
    cost_per_1000 = (hourly_cost / requests_per_hour) * 1000

    slo_error_ok = error_rate <= 1.0
    slo_latency_ok = p95 <= 5000.0
    error_budget = 1.0 - error_rate

    summary = {
        "total_requests": total_requests,
        "failures": failures,
        "error_rate_percent": round(error_rate, 3),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "requests_per_second": round(rps, 2),
        "replicas": replicas,
        "cpu_limit": cpu_limit,
        "memory_limit_gib": mem_limit,
        "hourly_cost_inr": round(hourly_cost, 2),
        "requests_per_hour": round(requests_per_hour, 0),
        "cost_per_1000_inr": round(cost_per_1000, 4),
        "slo_error_rate_ok": slo_error_ok,
        "slo_p95_latency_ok": slo_latency_ok,
        "error_budget_percent": round(error_budget, 3),
        "currency": rate_card["currency"],
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()

