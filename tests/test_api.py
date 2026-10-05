"""HTTP contract: auth, validation, rate limiting, caching, error mapping."""

import pytest
from fastapi.testclient import TestClient

from app.agent.graph import RAGAgent
from app.llm.client import LLMUnavailableError
from app.main import create_app
from tests.fakes import FakeLLM, FakeRetriever

AUTH = {"X-API-Key": "test-key"}


def make_client(settings, llm=None) -> TestClient:
    agent = RAGAgent(llm or FakeLLM(route="direct"), FakeRetriever(), settings)
    return TestClient(create_app(settings, agent=agent))


def test_health_needs_no_auth(settings):
    with make_client(settings) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json()["status"] == "ready"


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_query_requires_valid_api_key(settings, headers):
    with make_client(settings) as client:
        resp = client.post("/query", json={"question": "hi"}, headers=headers)
        assert resp.status_code == 401


def test_query_success_contract(settings):
    with make_client(settings) as client:
        resp = client.post("/query", json={"question": "Explain overfitting."}, headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "direct" and body["answer"] == "A general answer."
    assert body["usage"]["llm_calls"] == 2 and body["cached"] is False
    assert resp.headers["X-Request-ID"] == body["request_id"]


@pytest.mark.parametrize("question", ["", "x" * 2001])
def test_question_length_is_validated(settings, question):
    with make_client(settings) as client:
        assert client.post("/query", json={"question": question}, headers=AUTH).status_code == 422


def test_rate_limit_returns_429_with_retry_after(settings):
    settings.rate_limit_per_minute = 2
    with make_client(settings) as client:
        codes = [client.post("/query", json={"question": f"q{i}"}, headers=AUTH).status_code for i in range(3)]
        assert codes == [200, 200, 429]
        limited = client.post("/query", json={"question": "q9"}, headers=AUTH)
        assert int(limited.headers["Retry-After"]) >= 1
        # quotas are per key
        assert client.post("/query", json={"question": "q"}, headers={"X-API-Key": "second-key"}).status_code == 200


def test_repeat_question_is_served_from_cache(settings):
    llm = FakeLLM(route="direct")
    with make_client(settings, llm) as client:
        first = client.post("/query", json={"question": "Explain  Overfitting"}, headers=AUTH).json()
        second = client.post("/query", json={"question": "explain overfitting"}, headers=AUTH).json()
    assert second["cached"] is True and second["answer"] == first["answer"]
    assert llm.calls.count("generate_direct") == 1


def test_llm_outage_maps_to_503(settings):
    class DownLLM(FakeLLM):
        async def structured(self, *args, **kwargs):
            raise LLMUnavailableError("router: APITimeoutError")

    with make_client(settings, DownLLM()) as client:
        resp = client.post("/query", json={"question": "hi"}, headers=AUTH)
    assert resp.status_code == 503 and "temporarily unavailable" in resp.json()["detail"]


def test_refuses_to_start_without_keys_when_auth_enabled(settings):
    settings.api_keys = settings.api_keys.__class__("")
    with pytest.raises(RuntimeError, match="API_KEYS"), make_client(settings):
        pass


def test_feedback_and_metrics(settings):
    with make_client(settings) as client:
        client.post("/query", json={"question": "hello"}, headers=AUTH)
        assert client.post("/feedback", json={"request_id": "abc", "rating": 1}, headers=AUTH).status_code == 202
        metrics = client.get("/metrics/").text
    assert 'rag_requests_total{route="direct",status="ok"}' in metrics
    assert "rag_feedback_total" in metrics
