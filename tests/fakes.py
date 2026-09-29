"""Deterministic stand-ins for the LLM and retriever, so agent control flow is testable offline."""

from types import SimpleNamespace

from app.agent.schemas import CriterionResult, CriticVerdict, RouteDecision
from app.llm.client import LLMCall, ModerationResult
from app.rag.retriever import RetrievedChunk


def verdict(grounded=True, relevant=True, complete=True, sufficient=True, feedback="",
            rewritten="") -> CriticVerdict:
    def c(ok: bool) -> CriterionResult:
        return CriterionResult(passed=ok, reason="ok" if ok else "failed")
    return CriticVerdict(grounded=c(grounded), relevant=c(relevant), complete=c(complete),
                         context_sufficient=sufficient, feedback=feedback, rewritten_query=rewritten)


def chunk(source="hr_benefits_policies", chunk_id=0, score=0.8, text="Employees get twenty days.") -> RetrievedChunk:
    return RetrievedChunk(source=source, chunk_id=chunk_id, text=text, score=score)


class FakeLLM:
    def __init__(self, route="rag", answers=None, verdicts=None, direct_answer="A general answer.",
                 flag_input=False, flag_output=False):
        self.route = route
        self.answers = list(answers or [])
        self.verdicts = list(verdicts or [])
        self.direct_answer = direct_answer
        self.flag_input, self.flag_output = flag_input, flag_output
        self.calls: list[str] = []
        self.messages: dict[str, list] = {}
        self._moderations = 0

    def _call(self, stage: str) -> LLMCall:
        return LLMCall(stage=stage, model="fake", input_tokens=100, output_tokens=20, latency_ms=1.0,
                       cost_usd=0.0001)

    async def complete(self, stage, messages):
        self.calls.append(stage)
        self.messages.setdefault(stage, []).append(messages)
        text = self.direct_answer if stage == "generate_direct" else self.answers.pop(0)
        return text, self._call(stage)

    async def structured(self, stage, messages, schema, temperature=0.0):
        self.calls.append(stage)
        self.messages.setdefault(stage, []).append(messages)
        if stage == "router":
            return RouteDecision(route=self.route, reason="test"), self._call(stage)
        return self.verdicts.pop(0), self._call(stage)

    async def moderate(self, text):
        self.calls.append("moderation")
        self._moderations += 1
        flagged = self.flag_input if self._moderations == 1 else self.flag_output
        return ModerationResult(flagged, ["violence"] if flagged else [])


class FakeRetriever:
    """Returns `responses[i]` on the i-th search (last one repeats)."""

    def __init__(self, *responses: list[RetrievedChunk]):
        self.responses = list(responses) or [[chunk()]]
        self.queries: list[tuple[str, int]] = []
        self.store = SimpleNamespace(manifest={"index_hash": "test", "embedding_model": "fake"})

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        self.queries.append((query, k))
        return self.responses[min(len(self.queries) - 1, len(self.responses) - 1)][:k]
