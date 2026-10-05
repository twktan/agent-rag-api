"""Optional MLflow Tracing: one trace per request, one span per agent node and LLM call.

Enabled only when MLFLOW_TRACKING_URI is set (local docker-compose stack, offline eval).
In production on Cloud Run we rely on structured logs -> Cloud Logging instead, so tracing
is a no-op there and can never fail or slow down a request.
"""

import logging
from contextlib import contextmanager, suppress
from typing import Any

logger = logging.getLogger(__name__)
_mlflow = None


class _NoopSpan:
    def set_inputs(self, *_: Any) -> None: ...
    def set_outputs(self, *_: Any) -> None: ...
    def set_attributes(self, *_: Any) -> None: ...
    def set_attribute(self, *_: Any) -> None: ...


def setup_tracing(tracking_uri: str | None, experiment: str) -> bool:
    global _mlflow
    if not tracking_uri:
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.config.enable_async_logging(True)  # export spans off the request path
        _mlflow = mlflow
        logger.info("MLflow tracing enabled", extra={"fields": {"tracking_uri": tracking_uri}})
        return True
    except Exception as exc:  # tracing must never take the service down
        logger.warning("MLflow tracing disabled", extra={"fields": {"error": repr(exc)}})
        return False


def enabled() -> bool:
    return _mlflow is not None


@contextmanager
def span(name: str, span_type: str = "CHAIN", inputs: dict | None = None):
    if _mlflow is None:
        yield _NoopSpan()
        return
    with _mlflow.start_span(name=name, span_type=span_type) as s:
        if inputs is not None:
            s.set_inputs(inputs)
        yield s


def flush() -> None:
    if _mlflow is not None:
        with suppress(Exception):  # best effort on shutdown
            _mlflow.flush_trace_async_logging()
