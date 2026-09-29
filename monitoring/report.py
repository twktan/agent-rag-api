"""Daily health report from the API's structured logs.

Production (Cloud Logging):
    gcloud logging read 'resource.type="cloud_run_revision" AND
        resource.labels.service_name="fastapi-rag" AND
        (jsonPayload.event="query_completed" OR jsonPayload.event="llm_error" OR jsonPayload.event="feedback")' \
        --freshness=7d --format=json > logs.json
    python monitoring/report.py logs.json

Local (docker compose / uvicorn stdout):
    docker compose -f monitoring/docker-compose.yml logs api --no-log-prefix | python monitoring/report.py -
"""

import json
import statistics
import sys
from collections import Counter, defaultdict

DRIFT_THRESHOLD = 0.60  # median top-1 similarity; the offline eval median is ~0.70


def load_events(path: str) -> list[dict]:
    if path == "-":
        raw = sys.stdin.read().strip()
    else:
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
    if raw.startswith("["):  # gcloud --format=json: list of LogEntry
        return [dict(e.get("jsonPayload", {}), time=e.get("timestamp", "")) for e in json.loads(raw)]
    events = []
    for line in raw.splitlines():  # JSON lines straight from the app
        line = line.strip()
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def rate(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "-"


def day_report(events: list[dict]) -> dict:
    queries = [e for e in events if e.get("event") == "query_completed"]
    fresh = [e for e in queries if not e.get("cached")]
    rag = [e for e in fresh if e.get("route") == "rag"]
    top1 = [e["retrieval_top1"] for e in rag if e.get("retrieval_top1") is not None]
    feedback = Counter(e.get("rating") for e in events if e.get("event") == "feedback")
    return {
        "queries": len(queries),
        "route_mix": dict(Counter(e.get("route") for e in queries)),
        "cache_hit": rate(len(queries) - len(fresh), len(queries)),
        "p50_ms": pct([e["latency_ms"] for e in fresh], 0.5),
        "p95_ms": pct([e["latency_ms"] for e in fresh], 0.95),
        "llm_errors": sum(e.get("event") == "llm_error" for e in events),
        "blocked": rate(sum(e.get("route") == "refuse" for e in queries), len(queries)),
        "refined": rate(sum(e.get("refinements", 0) > 0 for e in rag), len(rag)),
        "critic_fail": rate(sum(e.get("quality_passed") is False for e in rag), len(rag)),
        "abstained": rate(sum(bool(e.get("abstained")) for e in rag), len(rag)),
        "cost_usd": round(sum(e.get("cost_usd", 0) or 0 for e in fresh), 4),
        "median_top1": round(statistics.median(top1), 3) if top1 else None,
        "feedback": f"+{feedback.get(1, 0)} / -{feedback.get(-1, 0)}",
    }


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    by_day: dict[str, list[dict]] = defaultdict(list)
    for e in load_events(sys.argv[1]):
        by_day[str(e.get("time", ""))[:10]].append(e)
    cols = ["queries", "route_mix", "cache_hit", "p50_ms", "p95_ms", "llm_errors", "blocked", "refined",
            "critic_fail", "abstained", "cost_usd", "median_top1", "feedback"]
    print("| day | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    warnings = []
    for day in sorted(by_day):
        r = day_report(by_day[day])
        print(f"| {day} | " + " | ".join(str(r[c]) for c in cols) + " |")
        if r["median_top1"] is not None and r["median_top1"] < DRIFT_THRESHOLD:
            warnings.append(f"{day}: retrieval drift (median top-1 {r['median_top1']} < {DRIFT_THRESHOLD})")
        if r["llm_errors"]:
            warnings.append(f"{day}: {r['llm_errors']} LLM errors")
    for w in warnings:
        print(f"WARNING {w}")


if __name__ == "__main__":
    main()
