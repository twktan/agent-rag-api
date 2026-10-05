"""Deterministic answer checks and the refinement-loop control policy.

The LLM critic judges *quality*; this module decides *what to do about it*. Keeping the
loop policy in plain code makes termination guaranteed and unit-testable.
"""

import re

from app.agent.state import Attempt
from app.prompts.common import is_abstention

_CITATION = re.compile(r"\[([A-Za-z0-9_\-]+)\]")


def extract_citations(answer: str) -> list[str]:
    seen: dict[str, None] = {}
    for source in _CITATION.findall(answer):
        seen.setdefault(source, None)
    return list(seen)


def deterministic_checks(answer: str, docs: list[dict]) -> tuple[dict[str, bool], dict[str, str]]:
    """Free checks that run before (and can skip) the LLM critic call."""
    if not answer.strip():
        return {"non_empty": False}, {"non_empty": "The answer is empty."}
    if is_abstention(answer):
        return {"citations": True}, {}
    cited = extract_citations(answer)
    valid = sorted({d["source"] for d in docs})
    if not cited:
        return ({"citations": False},
                {"citations": f"Cite supporting documents as [id]; valid ids: {', '.join(valid)}."})
    invalid = sorted(set(cited) - set(valid))
    if invalid:
        return ({"citations": False},
                {"citations": f"These cited ids are not in the documents: {', '.join(invalid)}. "
                              f"Use only: {', '.join(valid)}."})
    return {"citations": True}, {}


def score(attempt: Attempt) -> int:
    return sum(attempt["checks"].values())


def decide(attempts: list[Attempt], retrievals: int, max_refinements: int, max_retrievals: int = 2) -> str:
    """Next step after critiquing the latest attempt: accept | revise | re_retrieve | abstain | stop."""
    last = attempts[-1]
    budget_left = len(attempts) - 1 < max_refinements
    can_retrieve = budget_left and retrievals < max_retrievals
    if last["passed"]:
        # A correct abstention on thin context gets one corrective retrieval before we give up.
        if last["abstained"] and not last["context_sufficient"] and can_retrieve:
            return "re_retrieve"
        return "accept"
    if not budget_left:
        return "stop"
    if len(attempts) >= 2 and score(last) <= score(attempts[-2]) and last["context_sufficient"]:
        return "stop"  # revising is not helping; don't burn more tokens
    if not last["context_sufficient"]:
        return "re_retrieve" if can_retrieve else "abstain"
    return "revise"


def best_attempt(attempts: list[Attempt]) -> Attempt:
    """Prefer grounded answers, then more passed checks, then the most recent."""
    ranked = sorted(enumerate(attempts),
                    key=lambda ia: (ia[1]["checks"].get("grounded", False), score(ia[1]), ia[0]))
    return ranked[-1][1]


def cited_sources(attempt: Attempt) -> list[dict]:
    """Citations in answer order, each with its best-scoring retrieved chunk."""
    best: dict[str, dict] = {}
    for d in attempt["docs"]:
        if d["source"] not in best or d["score"] > best[d["source"]]["score"]:
            best[d["source"]] = d
    return [{"source": s, "chunk_id": best[s]["chunk_id"], "score": best[s]["score"]}
            for s in extract_citations(attempt["answer"]) if s in best]
