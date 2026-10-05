"""LangGraph state. Keys with reducers can be written by parallel nodes in the same step."""

import operator
from typing import Annotated, TypedDict


def merge_timings(left: dict | None, right: dict | None) -> dict:
    """Sum per-stage latencies, so looping stages (generate, critique) report total time."""
    merged = dict(left or {})
    for stage, ms in (right or {}).items():
        merged[stage] = round(merged.get(stage, 0.0) + ms, 1)
    return merged


class Attempt(TypedDict):
    answer: str
    search_query: str
    docs: list[dict]
    checks: dict[str, bool]
    reasons: dict[str, str]
    feedback: str
    context_sufficient: bool
    abstained: bool
    passed: bool


class AgentState(TypedDict, total=False):
    request_id: str
    query: str
    # guardrails (input_guard runs in parallel with route)
    input_flags: list[str]
    blocked_reason: str | None
    # routing
    router_route: str
    route_reason: str
    route_decision: str  # guard + router verdict: what routing accuracy is measured on
    route: str  # path actually taken (may change if RAG finds no relevant context)
    # retrieval
    search_query: str
    top_k: int
    retrievals: int
    docs: list[dict]
    retrieval_top1: float | None
    context_flags: list[str]
    # generation + reflection (RAG only)
    draft: str | None
    feedback: str | None
    attempts: Annotated[list[Attempt], operator.add]
    decision: str | None
    # output
    answer: str
    sources: list[dict]
    quality: dict
    output_flags: list[str]
    # telemetry
    llm_calls: Annotated[list[dict], operator.add]
    timings_ms: Annotated[dict, merge_timings]
