"""
load_test.py — API latency benchmark and smoke test
====================================================
Tests the /score and /score/batch endpoints against a running API server.

Usage
-----
    # Start the API first:
    uvicorn api.app:app --port 8000

    # Then in another terminal:
    python scripts/load_test.py                      # default: 200 single requests
    python scripts/load_test.py --n 500 --batch 32  # 500 singles + batch of 32
    python scripts/load_test.py --host http://my-server:8000
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from typing import Any

try:
    import httpx
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

BASE_URL = "http://localhost:8000"

# ── Sample transaction generator ──────────────────────────────────────────────

def _random_tx(tx_id: int | None = None) -> dict[str, Any]:
    return {
        "TransactionID": tx_id or random.randint(1_000_000, 9_999_999),
        "TransactionDT": random.uniform(0, 15_000_000),
        "TransactionAmt": round(random.uniform(1.0, 5000.0), 2),
        "card1": random.randint(1000, 20000),
        "card4": random.choice(["visa", "mastercard", "discover", "american express"]),
        "card6": random.choice(["debit", "credit"]),
        "ProductCD": random.choice(["W", "H", "C", "S", "R"]),
        **{f"V{i}": round(random.gauss(0, 1), 4) for i in range(1, 29)},
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_health(client: httpx.Client) -> None:
    r = client.get(f"{BASE_URL}/health", timeout=5)
    r.raise_for_status()
    r2 = client.get(f"{BASE_URL}/ready", timeout=5)
    if r2.status_code == 503:
        print("⚠  /ready returned 503 — models may not be loaded (run training pipeline first).")
    else:
        r2.raise_for_status()
        print("✓  Server is healthy and ready.\n")


def _percentile(data: list[float], p: int) -> float:
    data_sorted = sorted(data)
    idx = int(len(data_sorted) * p / 100)
    return data_sorted[min(idx, len(data_sorted) - 1)]


# ── Single-request benchmark ──────────────────────────────────────────────────

def bench_single(client: httpx.Client, n: int) -> list[float]:
    print(f"── Single /score  ({n} requests) ──────────────────────")
    latencies: list[float] = []
    errors = 0

    for i in range(n):
        tx = _random_tx(i + 1)
        t0 = time.perf_counter()
        try:
            r = client.post(f"{BASE_URL}/score", json=tx, timeout=10)
            elapsed = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                latencies.append(elapsed)
                data = r.json()
                if i == 0:  # print first response as example
                    print(f"  Sample response: p_fraud={data['fraud_probability']:.4f}  "
                          f"flag={data['fraud_flag']}  server_ms={data['latency_ms']}")
            else:
                errors += 1
                if i < 3:
                    print(f"  ERROR {r.status_code}: {r.text[:120]}")
        except httpx.RequestError as exc:
            errors += 1
            print(f"  Request error: {exc}")

    if latencies:
        p = _percentile
        print(f"\n  Requests   : {n}  (errors={errors})")
        print(f"  Latency ms : p50={p(latencies,50):.1f}  p90={p(latencies,90):.1f}  "
              f"p95={p(latencies,95):.1f}  p99={p(latencies,99):.1f}  "
              f"mean={statistics.mean(latencies):.1f}  max={max(latencies):.1f}")
        slow = sum(1 for l in latencies if l > 100)
        print(f"  > 100 ms   : {slow} ({slow/len(latencies)*100:.1f}%)")
    return latencies


# ── Batch benchmark ───────────────────────────────────────────────────────────

def bench_batch(client: httpx.Client, batch_size: int, repeats: int = 20) -> None:
    print(f"\n── Batch /score/batch  (batch_size={batch_size}, {repeats} calls) ──")
    latencies: list[float] = []

    for _ in range(repeats):
        payload = {
            "transactions": [_random_tx() for _ in range(batch_size)],
            "threshold": 0.5,
        }
        t0 = time.perf_counter()
        r = client.post(f"{BASE_URL}/score/batch", json=payload, timeout=30)
        elapsed = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            latencies.append(elapsed)
        else:
            print(f"  ERROR {r.status_code}: {r.text[:120]}")

    if latencies:
        p = _percentile
        flagged = r.json()["results"]
        n_fraud = sum(1 for x in flagged if x["fraud_flag"])
        print(f"  Batch size : {batch_size}")
        print(f"  Total ms   : p50={p(latencies,50):.1f}  p90={p(latencies,90):.1f}  mean={statistics.mean(latencies):.1f}")
        print(f"  Per-tx ms  : mean={statistics.mean(latencies)/batch_size:.2f}")
        print(f"  Fraud flags: {n_fraud}/{batch_size} in last batch")


# ── Correctness smoke test ────────────────────────────────────────────────────

def smoke_test(client: httpx.Client) -> None:
    print("\n── Smoke tests ─────────────────────────────────────────")

    # Test 1: valid request
    tx = _random_tx(9999999)
    r = client.post(f"{BASE_URL}/score", json=tx, timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["fraud_flag"], bool)
    print("  ✓ Valid request → 200, probability in [0,1]")

    # Test 2: missing required field
    bad = {"TransactionID": 1, "TransactionDT": 0.0}  # missing TransactionAmt
    r2 = client.post(f"{BASE_URL}/score", json=bad, timeout=5)
    assert r2.status_code == 422, f"Expected 422, got {r2.status_code}"
    print("  ✓ Missing required field → 422 Unprocessable Entity")

    # Test 3: zero amount
    bad2 = {**_random_tx(), "TransactionAmt": 0.0}
    r3 = client.post(f"{BASE_URL}/score", json=bad2, timeout=5)
    assert r3.status_code == 422, f"Expected 422, got {r3.status_code}"
    print("  ✓ Zero amount → 422 Unprocessable Entity")

    # Test 4: determinism — same input → same output
    tx_fixed = _random_tx(42)
    r4a = client.post(f"{BASE_URL}/score", json=tx_fixed, timeout=5).json()
    r4b = client.post(f"{BASE_URL}/score", json=tx_fixed, timeout=5).json()
    assert r4a["fraud_probability"] == r4b["fraud_probability"], "Non-deterministic output!"
    print("  ✓ Determinism — same input produces identical output")

    # Test 5: metrics endpoint
    r5 = client.get(f"{BASE_URL}/metrics", timeout=5)
    assert r5.status_code == 200
    m = r5.json()
    assert "total_requests" in m
    print(f"  ✓ /metrics OK — {m['total_requests']} requests recorded")

    print("\n  All smoke tests passed. ✓")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud API load test")
    parser.add_argument("--host",  default=BASE_URL, help="API base URL")
    parser.add_argument("--n",     type=int, default=200, help="Number of single requests")
    parser.add_argument("--batch", type=int, default=32,  help="Batch size for batch benchmark")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke tests")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.host.rstrip("/")

    print(f"Fraud Detection API — Load Test")
    print(f"Target: {BASE_URL}\n")

    with httpx.Client() as client:
        _check_health(client)

        if not args.skip_smoke:
            smoke_test(client)

        bench_single(client, args.n)
        bench_batch(client, args.batch)

        # Print server-side metrics
        r = client.get(f"{BASE_URL}/metrics")
        if r.status_code == 200:
            print("\n── Server-side metrics ─────────────────────────────────")
            print(json.dumps(r.json(), indent=2))


if __name__ == "__main__":
    main()
