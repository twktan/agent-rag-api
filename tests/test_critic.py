from app.agent.critic import best_attempt, decide, deterministic_checks, extract_citations
from app.prompts.common import ABSTAIN_MESSAGE

DOCS = [{"source": "hr_benefits_policies", "chunk_id": 0, "score": 0.8, "text": "..."}]


def attempt(passed=True, grounded=True, sufficient=True, abstained=False, n_checks=4):
    checks = {"citations": True, "grounded": grounded, "relevant": True, "complete": passed}
    return {"answer": "x", "search_query": "q", "docs": DOCS, "checks": dict(list(checks.items())[:n_checks]),
            "reasons": {}, "feedback": "", "context_sufficient": sufficient, "abstained": abstained,
            "passed": passed}


def test_extract_citations_dedupes_in_order():
    assert extract_citations("A [b] then [a] and [b] again.") == ["b", "a"]


def test_deterministic_checks():
    assert deterministic_checks("", DOCS)[0] == {"non_empty": False}
    assert deterministic_checks("No cites.", DOCS)[0] == {"citations": False}
    assert deterministic_checks("Bad [made_up_doc].", DOCS)[0] == {"citations": False}
    assert deterministic_checks("Good [hr_benefits_policies].", DOCS)[0] == {"citations": True}
    assert deterministic_checks(ABSTAIN_MESSAGE, DOCS)[0] == {"citations": True}


def test_decide_policy():
    assert decide([attempt()], retrievals=1, max_refinements=2) == "accept"
    assert decide([attempt(passed=False)], 1, 2) == "revise"
    assert decide([attempt(passed=False, sufficient=False)], 1, 2) == "re_retrieve"
    assert decide([attempt(passed=False, sufficient=False)], 2, 2) == "abstain"
    assert decide([attempt(passed=False)] * 3, 1, 2) == "stop"  # budget exhausted
    assert decide([attempt(passed=False), attempt(passed=False)], 1, 2) == "stop"  # no improvement
    assert decide([attempt(abstained=True, sufficient=False)], 1, 2) == "re_retrieve"
    assert decide([attempt(abstained=True, sufficient=False)], 2, 2) == "accept"
    assert decide([attempt(passed=False)], 1, max_refinements=0) == "stop"


def test_best_attempt_prefers_grounded():
    ungrounded = attempt(passed=False, grounded=False)
    grounded = attempt(passed=False, grounded=True)
    grounded["answer"] = "grounded"
    assert best_attempt([grounded, ungrounded])["answer"] == "grounded"
