"""Single-agent LangGraph state machine: guard + route -> (RAG: retrieve -> generate -> critique
-> refine loop) | (DIRECT: generate) | (REFUSE) -> output guard.

The LLM makes the judgement calls (route, pass/fail per criterion, feedback, query rewrite);
code owns control flow and budgets (app/agent/critic.py::decide), so every request
terminates within a known worst-case number of LLM calls.

    python -m app.agent.graph     # print the graph as a Mermaid diagram
"""

import asyncio
import secrets
import time

from langgraph.graph import END, START, StateGraph

from app.agent.critic import best_attempt, cited_sources, decide, deterministic_checks
from app.agent.schemas import CriticVerdict, RouteDecision
from app.agent.state import AgentState, Attempt
from app.config import Settings
from app.guardrails.input_guard import detect_injection, filter_injected_chunks
from app.guardrails.output_guard import leaked_canary, redact_pii
from app.llm.client import LLMOutputError
from app.observability import tracing
from app.prompts.common import ABSTAIN_MESSAGE, REFUSAL_MESSAGE, is_abstention
from app.prompts.critic import critic_messages
from app.prompts.direct import direct_messages
from app.prompts.rag import rag_messages
from app.prompts.router import router_messages

_SPAN_TYPES = {"route": "AGENT", "retrieve": "RETRIEVER", "critique": "AGENT", "input_guard": "GUARDRAIL",
               "finalize": "GUARDRAIL"}
_HEAVY_KEYS = ("llm_calls", "docs", "attempts", "timings_ms")


def _node(name: str, fn):
    """Wrap a node with per-stage timing and an (optional) MLflow span."""
    async def wrapper(state: AgentState) -> dict:
        start = time.perf_counter()
        with tracing.span(f"node.{name}", _SPAN_TYPES.get(name, "CHAIN"),
                          {"query": state.get("search_query") or state.get("query")}) as span:
            update = dict(await fn(state) or {})
            span.set_outputs({k: v for k, v in update.items() if k not in _HEAVY_KEYS})
        update["timings_ms"] = {name: round((time.perf_counter() - start) * 1000, 1)}
        return update
    wrapper.__name__ = name
    return wrapper


class RAGAgent:
    def __init__(self, llm, retriever, settings: Settings):
        self.llm = llm
        self.retriever = retriever
        self.settings = settings
        # Unique per process; if it ever shows up in an answer, the system prompt leaked.
        self.canary = f"CANARY-{secrets.token_hex(6)}"
        self.graph = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(AgentState)
        nodes = {
            "input_guard": self._input_guard, "route": self._route, "dispatch": self._dispatch,
            "retrieve": self._retrieve, "generate_rag": self._generate_rag, "critique": self._critique,
            "abstain": self._abstain, "reroute_direct": self._reroute_direct,
            "generate_direct": self._generate_direct, "refuse": self._refuse, "finalize": self._finalize,
        }
        for name, fn in nodes.items():
            g.add_node(name, _node(name, fn))

        # Guardrail and router are independent, so they run in parallel and join at dispatch.
        g.add_edge(START, "input_guard")
        g.add_edge(START, "route")
        g.add_edge(["input_guard", "route"], "dispatch")
        g.add_conditional_edges("dispatch", lambda s: s["route"],
                                {"rag": "retrieve", "direct": "generate_direct", "refuse": "refuse"})
        g.add_conditional_edges("retrieve", self._after_retrieve,
                                {"generate": "generate_rag", "abstain": "abstain",
                                 "reroute_direct": "reroute_direct"})
        g.add_edge("generate_rag", "critique")
        g.add_conditional_edges("critique", lambda s: s["decision"],
                                {"accept": "finalize", "stop": "finalize", "revise": "generate_rag",
                                 "re_retrieve": "retrieve", "abstain": "abstain"})
        g.add_edge("reroute_direct", "generate_direct")
        for terminal in ("generate_direct", "refuse", "abstain"):
            g.add_edge(terminal, "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    async def run(self, query: str, request_id: str = "") -> AgentState:
        initial: AgentState = {"request_id": request_id, "query": query, "attempts": [], "llm_calls": [],
                               "timings_ms": {}, "context_flags": [], "output_flags": []}
        with tracing.span("agent.run", "AGENT", {"query": query, "request_id": request_id}) as span:
            final = await self.graph.ainvoke(initial)
            span.set_outputs({"answer": final["answer"], "route": final["route"],
                              "quality": final.get("quality")})
        return final

    # ------------------------------------------------------------------ nodes
    async def _input_guard(self, state: AgentState) -> dict:
        flags = detect_injection(state["query"])
        if flags:
            return {"input_flags": flags, "blocked_reason": "prompt_injection"}
        if self.settings.moderation_enabled:
            result = await self.llm.moderate(state["query"])
            if result.flagged:
                return {"input_flags": result.categories, "blocked_reason": "moderation"}
        return {"input_flags": [], "blocked_reason": None}

    async def _route(self, state: AgentState) -> dict:
        try:
            decision, call = await self.llm.structured("router", router_messages(state["query"]),
                                                       RouteDecision)
        except LLMOutputError:
            # Safe default: RAG is grounded and abstains when the docs don't cover the question.
            return {"router_route": "rag", "route_reason": "router_output_unparseable"}
        return {"router_route": decision.route, "route_reason": decision.reason, "llm_calls": [call.to_dict()]}

    async def _dispatch(self, state: AgentState) -> dict:
        blocked = state.get("blocked_reason")
        route = "refuse" if blocked else state["router_route"]
        if route == "refuse" and not blocked:
            blocked = "router"
        return {"route_decision": route, "route": route, "blocked_reason": blocked,
                "search_query": state["query"], "top_k": self.settings.top_k, "retrievals": 0}

    async def _retrieve(self, state: AgentState) -> dict:
        hits = await asyncio.to_thread(self.retriever.search, state["search_query"], state["top_k"])
        docs = [h.to_dict() for h in hits if h.score >= self.settings.min_similarity]
        docs, suspicious = filter_injected_chunks(docs)
        update = {"docs": docs, "retrievals": state["retrievals"] + 1, "draft": None, "feedback": None}
        if state["retrievals"] == 0:
            update["retrieval_top1"] = hits[0].score if hits else None
        if suspicious:
            update["context_flags"] = state.get("context_flags", []) + [
                f"indirect_injection:{c['source']}#{c['chunk_id']}" for c in suspicious]
        return update

    def _after_retrieve(self, state: AgentState) -> str:
        if state["docs"]:
            return "generate"
        # Nothing relevant on the first search: most likely a router false positive.
        if state["retrievals"] == 1 and not state.get("attempts"):
            return "reroute_direct"
        return "abstain"

    async def _generate_rag(self, state: AgentState) -> dict:
        messages = rag_messages(state["query"], state["docs"], self.canary,
                                previous_answer=state.get("draft"), feedback=state.get("feedback"))
        text, call = await self.llm.complete("generate_rag", messages)
        return {"draft": text.strip(), "llm_calls": [call.to_dict()]}

    async def _critique(self, state: AgentState) -> dict:
        answer, docs = state["draft"], state["docs"]
        checks, reasons = deterministic_checks(answer, docs)
        context_sufficient, rewritten, calls = True, "", []
        if all(checks.values()):  # only pay for the LLM critic if the free checks pass
            verdict, call = await self.llm.structured("critic", critic_messages(state["query"], docs, answer),
                                                      CriticVerdict)
            calls.append(call.to_dict())
            for name in ("grounded", "relevant", "complete"):
                result = getattr(verdict, name)
                checks[name] = result.passed
                if not result.passed:
                    reasons[name] = result.reason
            context_sufficient, rewritten = verdict.context_sufficient, verdict.rewritten_query.strip()
            feedback = verdict.feedback
        else:
            feedback = " ".join(reasons.values())
        attempt = Attempt(answer=answer, search_query=state["search_query"], docs=docs, checks=checks,
                          reasons=reasons, feedback=feedback, context_sufficient=context_sufficient,
                          abstained=is_abstention(answer), passed=all(checks.values()))
        decision = decide(state["attempts"] + [attempt], state["retrievals"], self.settings.max_refinements)
        update = {"attempts": [attempt], "decision": decision, "feedback": feedback, "llm_calls": calls}
        if decision == "re_retrieve":
            update["search_query"] = rewritten or state["query"]
            update["top_k"] = min(state["top_k"] * 2, self.settings.max_top_k)
        return update

    async def _abstain(self, state: AgentState) -> dict:
        return {"answer": ABSTAIN_MESSAGE, "decision": "abstain"}

    async def _reroute_direct(self, state: AgentState) -> dict:
        return {"route": "direct", "decision": "rerouted_no_context"}

    async def _generate_direct(self, state: AgentState) -> dict:
        text, call = await self.llm.complete("generate_direct", direct_messages(state["query"], self.canary))
        return {"answer": text.strip(), "llm_calls": [call.to_dict()]}

    async def _refuse(self, state: AgentState) -> dict:
        return {"answer": REFUSAL_MESSAGE}

    async def _finalize(self, state: AgentState) -> dict:
        route, attempts, decision = state["route"], state.get("attempts", []), state.get("decision")
        answer, sources = state.get("answer", ""), []
        quality = {"checked": route == "rag", "passed": None, "iterations": len(attempts),
                   "refinements": max(len(attempts) - 1, 0), "failed_criteria": [], "decision": decision}

        if route == "rag" and decision in ("accept", "stop") and attempts:
            chosen = attempts[-1] if decision == "accept" else best_attempt(attempts)
            quality["failed_criteria"] = sorted(k for k, ok in chosen["checks"].items() if not ok)
            quality["passed"] = chosen["passed"]
            if decision == "stop" and not chosen["checks"].get("grounded", False):
                answer = ABSTAIN_MESSAGE  # never return an answer the critic found ungrounded
                quality["fallback"] = "abstain_ungrounded"
            else:
                answer, sources = chosen["answer"], cited_sources(chosen)

        flags = list(state.get("output_flags", []))
        if leaked_canary(answer, self.canary):
            answer, sources = REFUSAL_MESSAGE, []
            flags.append("system_prompt_leak")
        answer, pii = redact_pii(answer)
        flags += [f"pii_redacted:{kind}" for kind in pii]
        # RAG answers are constrained to vetted internal docs and already critiqued; open-ended
        # direct answers get an output moderation pass.
        if (route == "direct" and self.settings.output_moderation_enabled
                and (await self.llm.moderate(answer)).flagged):
            answer = REFUSAL_MESSAGE
            flags.append("output_moderation")
        return {"answer": answer, "sources": sources, "quality": quality, "output_flags": flags}


if __name__ == "__main__":
    from types import SimpleNamespace

    stub = RAGAgent(llm=None, retriever=None, settings=SimpleNamespace())
    print(stub.graph.get_graph().draw_mermaid())
