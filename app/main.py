"""FastAPI service.

    uvicorn app.main:create_app --factory --port 8080
"""

import asyncio
import hashlib
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from app import __version__
from app.agent.graph import RAGAgent
from app.agent.summary import summarize_run, usage_totals
from app.cache import TTLCache, normalize
from app.config import Settings, get_settings
from app.llm.client import LLMClient, LLMOutputError, LLMUnavailableError
from app.observability import tracing
from app.observability.logging import configure_logging, log_event, request_context
from app.observability.metrics import CACHE_HITS, FEEDBACK, REQUESTS, record_query
from app.prompts import PROMPT_VERSION
from app.rag.retriever import Retriever
from app.schemas import ErrorResponse, FeedbackRequest, QueryRequest, QueryResponse
from app.security import RateLimiter, make_api_key_dependency

logger = logging.getLogger("app")

DESCRIPTION = """
Internal AI assistant for employees of the fictional company **Trevor Tan Incorporated (TTI)**.

A single LangGraph agent handles every question:
- **rag**: TTI-specific questions. It retrieves from the internal docs, writes an answer with
  `[source]` citations, and a critic checks it for grounding, relevance, completeness and citations.
  If the answer fails, the agent revises it or searches again, up to 2 refinements.
- **direct**: general questions are answered by the LLM directly, with no refinement loop.
- **refuse**: prompt injection, harmful requests or requests for personal data are refused by
  layered guardrails.

**Authentication:** click **Authorize** and enter your `X-API-Key`. Keys are issued on request.

Try: *"How many vacation days do I get per year?"*, *"Which CI/CD system do we use?"*,
*"Explain the bias-variance trade-off."*
"""

_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_QUIET_PATHS = ("/health", "/metrics")


def create_app(settings: Settings | None = None, agent: RAGAgent | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.google_cloud_project)
    limiter = RateLimiter(settings.rate_limit_per_minute, settings.daily_quota_per_key,
                          settings.global_daily_quota)
    cache = TTLCache(settings.cache_max_entries, settings.cache_ttl_s)
    require_api_key = make_api_key_dependency(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.auth_enabled and not settings.api_key_set:
            raise RuntimeError("AUTH_ENABLED=true but API_KEYS is empty; refusing to start unauthenticated")
        if settings.env == "prod" and not settings.auth_enabled:
            raise RuntimeError("Refusing to run in prod with authentication disabled")
        tracing.setup_tracing(settings.mlflow_tracking_uri, settings.mlflow_experiment)
        if agent is not None:
            app.state.agent = agent
        else:
            retriever = await asyncio.to_thread(Retriever.from_index_dir, settings.index_dir,
                                                settings.min_similarity, settings.embed_threads)
            app.state.agent = RAGAgent(LLMClient(settings), retriever, settings)
        manifest = getattr(getattr(app.state.agent.retriever, "store", None), "manifest", {})
        log_event(logger, "startup_complete", version=__version__, prompt_version=PROMPT_VERSION,
                  llm_model=settings.llm_model, index_hash=manifest.get("index_hash"),
                  embedding_model=manifest.get("embedding_model"))
        yield
        tracing.flush()

    app = FastAPI(title="Enterprise Agentic RAG API", version=__version__, description=DESCRIPTION,
                  lifespan=lifespan)

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex[:16]
        trace_id = request.headers.get("X-Cloud-Trace-Context", "").split("/")[0] or None
        request.state.request_id = request_id
        token = request_context.set({"request_id": request_id, "trace_id": trace_id})
        start = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            if request.url.path not in _QUIET_PATHS:
                log_event(logger, "http_request", method=request.method, path=request.url.path,
                          status=response.status_code,
                          latency_ms=round((time.perf_counter() - start) * 1000, 1))
            return response
        finally:
            request_context.reset(token)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        logger.exception("unhandled_error", extra={"fields": {"event": "unhandled_error",
                                                              "path": request.url.path}})
        return JSONResponse(status_code=500, content={
            "detail": "Internal server error", "request_id": getattr(request.state, "request_id", None)})

    @app.get("/health", tags=["ops"], summary="Liveness probe")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"], summary="Readiness: index and agent loaded")
    async def ready(request: Request) -> dict:
        agent_ = getattr(request.app.state, "agent", None)
        if agent_ is None:
            raise HTTPException(503, "Agent not loaded")
        manifest = getattr(getattr(agent_.retriever, "store", None), "manifest", {})
        return {"status": "ready", "version": __version__, "prompt_version": PROMPT_VERSION,
                "index_hash": manifest.get("index_hash"), "embedding_model": manifest.get("embedding_model")}

    @app.post("/query", response_model=QueryResponse, tags=["assistant"],
              summary="Ask the assistant a question",
              responses={401: {"model": ErrorResponse}, 422: {"description": "Invalid request"},
                         429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    async def query(req: QueryRequest, request: Request, caller: str = Depends(require_api_key)):
        limiter.check(caller)
        request_id = request.state.request_id
        cache_key = normalize(req.question)
        if (hit := cache.get(cache_key)) is not None:
            CACHE_HITS.inc()
            log_event(logger, "query_completed", cached=True, route=hit["route"], caller=caller,
                      latency_ms=0.0, cost_usd=0.0)
            return QueryResponse(**hit, request_id=request_id, cached=True)

        start = time.perf_counter()
        try:
            final = await request.app.state.agent.run(req.question, request_id)
        except (LLMUnavailableError, LLMOutputError) as exc:
            REQUESTS.labels(route="unknown", status="llm_error").inc()
            log_event(logger, "llm_error", level=logging.WARNING, error=str(exc), caller=caller)
            raise HTTPException(503, "The language model is temporarily unavailable. Please retry.") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        summary = summarize_run(final, latency_ms)
        _log_query(summary, caller, req.question, settings)
        if settings.metrics_enabled:
            record_query(summary)

        response = QueryResponse(
            request_id=request_id, answer=final["answer"], route=final["route"], sources=final["sources"],
            quality={k: final["quality"].get(k) for k in ("checked", "passed", "refinements",
                                                         "failed_criteria", "decision")},
            usage=usage_totals(final["llm_calls"]), latency_ms=round(latency_ms, 1))
        if final["route"] != "refuse":
            cache.set(cache_key, response.model_dump(exclude={"request_id", "cached"}))
        return response

    @app.post("/feedback", status_code=202, tags=["assistant"], summary="Rate an answer",
              responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}})
    async def feedback(fb: FeedbackRequest, caller: str = Depends(require_api_key)) -> dict:
        limiter.check(caller)
        FEEDBACK.labels(rating=str(fb.rating)).inc()
        log_event(logger, "feedback", feedback_request_id=fb.request_id, rating=fb.rating, caller=caller,
                  comment=fb.comment if settings.log_query_text else bool(fb.comment))
        return {"status": "accepted"}

    if settings.metrics_enabled:
        app.mount("/metrics", make_asgi_app())

    return app


def _log_query(summary: dict, caller: str, question: str, settings: Settings) -> None:
    fields = {k: v for k, v in summary.items() if k != "llm_call_log"}
    fields["llm_calls_detail"] = [{k: c[k] for k in ("stage", "input_tokens", "output_tokens",
                                                      "latency_ms", "cost_usd")}
                                  for c in summary["llm_call_log"]]
    fields["caller"] = caller
    fields["question_chars"] = len(question)
    if settings.log_query_text:
        fields["question"] = question[:500]
    else:  # stable id for grouping repeated questions without storing their text
        fields["question_sha"] = hashlib.sha256(normalize(question).encode()).hexdigest()[:12]
    log_event(logger, "query_completed", **fields)
