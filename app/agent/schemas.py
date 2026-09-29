"""Structured-output schemas the LLM must fill (OpenAI strict JSON schema: every field required)."""

from typing import Literal

from pydantic import BaseModel


class RouteDecision(BaseModel):
    route: Literal["rag", "direct", "refuse"]
    reason: str


class CriterionResult(BaseModel):
    passed: bool
    reason: str


class CriticVerdict(BaseModel):
    grounded: CriterionResult
    relevant: CriterionResult
    complete: CriterionResult
    context_sufficient: bool
    feedback: str
    rewritten_query: str
