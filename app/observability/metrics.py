"""Prometheus metrics (local/k8s). On Cloud Run the same signals come from log-based metrics."""

from prometheus_client import Counter, Histogram

_LATENCY_BUCKETS = (0.25, 0.5, 1, 2, 3, 5, 8, 13, 20, 30, 60)

REQUESTS = Counter("rag_requests_total", "Queries served", ["route", "status"])
LATENCY = Histogram("rag_request_latency_seconds", "End-to-end query latency", ["route"],
                    buckets=_LATENCY_BUCKETS)
STAGE_LATENCY = Histogram("rag_stage_latency_seconds", "Latency per agent stage", ["stage"],
                          buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16))
TOKENS = Counter("rag_llm_tokens_total", "LLM tokens", ["stage", "kind"])
COST = Counter("rag_llm_cost_usd_total", "Estimated LLM spend in USD", ["stage"])
GUARDRAIL_BLOCKS = Counter("rag_guardrail_blocks_total", "Requests blocked by guardrails", ["reason"])
REFINEMENTS = Histogram("rag_refinement_iterations", "Critic-driven refinements per RAG query",
                        buckets=(0, 1, 2, 3))
CRITIC_OUTCOME = Counter("rag_critic_outcome_total", "Final critic decision for RAG queries", ["decision"])
RETRIEVAL_TOP1 = Histogram("rag_retrieval_top1_similarity", "Top-1 cosine similarity (drift signal)",
                           buckets=(0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0))
CACHE_HITS = Counter("rag_cache_hits_total", "Responses served from cache")
RATE_LIMITED = Counter("rag_rate_limited_total", "Requests rejected by rate limiting", ["scope"])
FEEDBACK = Counter("rag_feedback_total", "User feedback", ["rating"])


def record_query(summary: dict) -> None:
    """Update metrics from the `query_completed` summary produced by the API layer."""
    route = summary["route"]
    REQUESTS.labels(route=route, status="ok").inc()
    LATENCY.labels(route=route).observe(summary["latency_ms"] / 1000)
    for stage, ms in summary.get("timings_ms", {}).items():
        STAGE_LATENCY.labels(stage=stage).observe(ms / 1000)
    for call in summary.get("llm_call_log", []):
        TOKENS.labels(stage=call["stage"], kind="input").inc(call["input_tokens"])
        TOKENS.labels(stage=call["stage"], kind="output").inc(call["output_tokens"])
        if call.get("cost_usd"):
            COST.labels(stage=call["stage"]).inc(call["cost_usd"])
    if summary.get("blocked_reason"):
        GUARDRAIL_BLOCKS.labels(reason=summary["blocked_reason"]).inc()
    if route == "rag":
        REFINEMENTS.observe(summary.get("refinements", 0))
        CRITIC_OUTCOME.labels(decision=summary.get("decision") or "none").inc()
    if summary.get("retrieval_top1") is not None:
        RETRIEVAL_TOP1.observe(summary["retrieval_top1"])
