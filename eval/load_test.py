"""Latency against a deployed endpoint (network + cold start included). Costs money; keep n small.

    RAG_API_KEY=... python -m eval.load_test --url https://<service>.run.app --n 30 --concurrency 2

Uses distinct questions so the response cache does not flatter the numbers; cached responses
are reported separately.
"""

import argparse
import asyncio
import os
import time
from collections import Counter

import httpx

from eval.common import RESULTS_DIR, load_golden, utc_now, write_json
from eval.metrics import latency_summary


async def run(url: str, api_key: str, n: int, concurrency: int) -> dict:
    questions = [i["question"] for i in load_golden("test") if i["category"] != "adversarial"][:n]
    headers = {"X-API-Key": api_key}
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async with httpx.AsyncClient(base_url=url.rstrip("/"), timeout=90) as client:
        start = time.perf_counter()
        await client.get("/health")
        first_ms = (time.perf_counter() - start) * 1000  # includes a cold start if the service was idle

        async def one(q: str) -> None:
            async with sem:
                t0 = time.perf_counter()
                resp = await client.post("/query", json={"question": q}, headers=headers)
                rec = {"status": resp.status_code, "client_ms": (time.perf_counter() - t0) * 1000}
                if resp.status_code == 200:
                    body = resp.json()
                    rec |= {"server_ms": body["latency_ms"], "route": body["route"], "cached": body["cached"]}
                results.append(rec)

        await asyncio.gather(*(one(q) for q in questions))

    fresh = [r for r in results if r["status"] == 200 and not r["cached"]]
    return {
        "url": url, "timestamp": utc_now(), "n": len(results), "concurrency": concurrency,
        "status_codes": dict(Counter(r["status"] for r in results)),
        "cached": sum(r.get("cached", False) for r in results),
        "first_request_ms": round(first_ms, 1),
        "client_latency_ms": latency_summary([r["client_ms"] for r in fresh]),
        "server_latency_ms": latency_summary([r["server_ms"] for r in fresh]),
        "client_latency_by_route_ms": {route: latency_summary([r["client_ms"] for r in fresh if r["route"] == route])
                                       for route in sorted({r["route"] for r in fresh})},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--concurrency", type=int, default=2)
    args = p.parse_args()
    api_key = os.environ.get("RAG_API_KEY") or exit("Set RAG_API_KEY")
    report = asyncio.run(run(args.url, api_key, args.n, args.concurrency))
    write_json(RESULTS_DIR / "load_test.json", report)
    c = report["client_latency_ms"]
    print(f"n={report['n']} status={report['status_codes']} cached={report['cached']} "
          f"first={report['first_request_ms']}ms p50={c.get('p50')}ms p95={c.get('p95')}ms")


if __name__ == "__main__":
    main()
