"""Async LLM access through LangChain's ChatOpenAI, with per-call token/cost/latency accounting.

Every call is bounded: request timeout, retry budget and max output tokens per stage, so a
single request has a hard worst-case cost.
"""

import logging
import time
from dataclasses import asdict, dataclass

import openai
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import Settings
from app.llm.pricing import cost_usd
from app.observability import tracing

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """The LLM provider failed after retries (timeout, rate limit, outage, bad credentials)."""


class LLMOutputError(RuntimeError):
    """The LLM returned output that does not match the requested schema."""


@dataclass
class LLMCall:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModerationResult:
    flagged: bool
    categories: list[str]


_PROVIDER_ERRORS = (openai.APIError, openai.APITimeoutError, openai.APIConnectionError, TimeoutError)


class LLMClient:
    def __init__(self, settings: Settings, model: str | None = None):
        self.model = model or settings.llm_model
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        self._max_tokens = {"router": settings.max_tokens_router, "critic": settings.max_tokens_critic}
        self._default_max_tokens = settings.max_tokens_answer
        self._common = {"model": self.model, "api_key": api_key, "timeout": settings.llm_timeout_s,
                        "max_retries": settings.llm_max_retries, "temperature": settings.llm_temperature}
        self._models: dict[str, ChatOpenAI] = {}
        self._moderation_model = settings.moderation_model
        self._openai = openai.AsyncOpenAI(api_key=api_key, timeout=10.0, max_retries=1)

    def _chat(self, stage: str, temperature: float | None = None) -> ChatOpenAI:
        key = f"{stage}:{temperature}"
        if key not in self._models:
            params = dict(self._common)
            if temperature is not None:
                params["temperature"] = temperature
            params["max_tokens"] = self._max_tokens.get(stage, self._default_max_tokens)
            self._models[key] = ChatOpenAI(**params)
        return self._models[key]

    def _record(self, stage: str, message, start: float) -> LLMCall:
        usage = getattr(message, "usage_metadata", None) or {}
        tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        return LLMCall(stage, self.model, tokens_in, tokens_out,
                       round((time.perf_counter() - start) * 1000, 1),
                       cost_usd(self.model, tokens_in, tokens_out))

    async def complete(self, stage: str, messages: list[tuple[str, str]]) -> tuple[str, LLMCall]:
        start = time.perf_counter()
        with tracing.span(f"llm.{stage}", span_type="LLM", inputs={"messages": messages}) as span:
            try:
                message = await self._chat(stage).ainvoke(messages)
            except _PROVIDER_ERRORS as exc:
                raise LLMUnavailableError(f"{stage}: {type(exc).__name__}") from exc
            call = self._record(stage, message, start)
            span.set_outputs({"text": message.content})
            span.set_attributes(call.to_dict())
        return str(message.content), call

    async def structured(self, stage: str, messages: list[tuple[str, str]], schema: type[BaseModel],
                         temperature: float | None = 0.0) -> tuple[BaseModel, LLMCall]:
        start = time.perf_counter()
        runnable = self._chat(stage, temperature).with_structured_output(
            schema, method="json_schema", include_raw=True, strict=True)
        with tracing.span(f"llm.{stage}", span_type="LLM", inputs={"messages": messages}) as span:
            try:
                out = await runnable.ainvoke(messages)
            except _PROVIDER_ERRORS as exc:
                raise LLMUnavailableError(f"{stage}: {type(exc).__name__}") from exc
            call = self._record(stage, out["raw"], start)
            if out.get("parsing_error") or out.get("parsed") is None:
                raise LLMOutputError(f"{stage}: could not parse {schema.__name__}")
            span.set_outputs(out["parsed"].model_dump())
            span.set_attributes(call.to_dict())
        return out["parsed"], call

    async def moderate(self, text: str) -> ModerationResult:
        """OpenAI moderation (free). Fails open: the router's refuse route is the backstop."""
        try:
            resp = await self._openai.moderations.create(model=self._moderation_model, input=text)
        except _PROVIDER_ERRORS as exc:
            logger.warning("moderation unavailable, failing open", extra={"fields": {"error": type(exc).__name__}})
            return ModerationResult(False, ["moderation_unavailable"])
        result = resp.results[0]
        flagged_categories = [k for k, v in result.categories.model_dump().items() if v]
        return ModerationResult(bool(result.flagged), flagged_categories)
