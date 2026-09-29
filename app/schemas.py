"""Public API contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_QUESTION_CHARS = 2000


class QueryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"question": "How many vacation days do I get per year?"},
                     {"question": "Explain the difference between supervised and unsupervised learning."}]})

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class Source(BaseModel):
    source: str = Field(description="Document id, cited in the answer as [source]")
    chunk_id: int
    score: float = Field(description="Cosine similarity of the best matching chunk")


class Quality(BaseModel):
    checked: bool = Field(description="True if the answer went through the critic (RAG route only)")
    passed: bool | None = Field(description="Critic verdict on the returned answer; null if not checked")
    refinements: int
    failed_criteria: list[str]
    decision: str | None = Field(description="accept | stop | abstain | rerouted_no_context")


class Usage(BaseModel):
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    route: Literal["rag", "direct", "refuse"]
    sources: list[Source]
    quality: Quality
    usage: Usage
    latency_ms: float
    cached: bool = False


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    rating: Literal[1, -1] = Field(description="1 = helpful, -1 = not helpful")
    comment: str | None = Field(default=None, max_length=1000)


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
