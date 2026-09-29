"""Flatten a finished agent run into one record for the API response, logs, metrics and eval."""

from app.agent.state import AgentState
from app.prompts import PROMPT_VERSION
from app.prompts.common import is_abstention


def usage_totals(llm_calls: list[dict]) -> dict:
    costs = [c["cost_usd"] for c in llm_calls if c.get("cost_usd") is not None]
    return {"llm_calls": len(llm_calls),
            "input_tokens": sum(c["input_tokens"] for c in llm_calls),
            "output_tokens": sum(c["output_tokens"] for c in llm_calls),
            "cost_usd": round(sum(costs), 6)}


def summarize_run(state: AgentState, latency_ms: float) -> dict:
    quality = state.get("quality", {})
    return {
        "request_id": state.get("request_id"),
        "route": state["route"],
        "route_decision": state.get("route_decision"),
        "route_reason": state.get("route_reason"),
        "blocked_reason": state.get("blocked_reason"),
        "input_flags": state.get("input_flags", []),
        "context_flags": state.get("context_flags", []),
        "output_flags": state.get("output_flags", []),
        "decision": quality.get("decision"),
        "abstained": is_abstention(state.get("answer", "")),
        "quality_passed": quality.get("passed"),
        "refinements": quality.get("refinements", 0),
        "failed_criteria": quality.get("failed_criteria", []),
        "retrievals": state.get("retrievals", 0),
        "retrieval_top1": state.get("retrieval_top1"),
        "sources": [s["source"] for s in state.get("sources", [])],
        "latency_ms": round(latency_ms, 1),
        "timings_ms": state.get("timings_ms", {}),
        "llm_call_log": state.get("llm_calls", []),
        **usage_totals(state.get("llm_calls", [])),
        "prompt_version": PROMPT_VERSION,
    }
