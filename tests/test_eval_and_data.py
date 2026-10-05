"""Metric correctness, eval-set integrity, and few-shot/eval leakage checks."""

import re
from pathlib import Path

import pytest

from app.prompts import critic as critic_prompt
from app.prompts.router import ROUTER_EXAMPLES
from eval.common import load_golden
from eval.metrics import (
    best_threshold,
    classification_report,
    first_relevant_rank,
    hit_at_k,
    mean_reciprocal_rank,
    wilson_interval,
)

DOCS = {p.stem for p in Path("data/docs").glob("*.txt")}


def test_ranking_metrics():
    ranks = [first_relevant_rank(["a", "b", "c"], {"b"}), first_relevant_rank(["x", "y"], {"b"}),
             first_relevant_rank(["b"], {"b"})]
    assert ranks == [2, None, 1]
    assert hit_at_k(ranks, 1) == pytest.approx(1 / 3)
    assert hit_at_k(ranks, 3) == pytest.approx(2 / 3)
    assert mean_reciprocal_rank(ranks) == pytest.approx((0.5 + 0 + 1) / 3)


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(45, 50)
    assert lo == pytest.approx(0.786, abs=1e-3) and hi == pytest.approx(0.957, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_classification_report():
    rep = classification_report(["rag", "rag", "direct", "refuse"], ["rag", "direct", "direct", "refuse"],
                                ["rag", "direct", "refuse"])
    assert rep["accuracy"]["value"] == 0.75
    assert rep["per_class"]["rag"] == {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}
    assert rep["confusion"]["rag"]["direct"] == 1


def test_best_threshold_separates_classes():
    assert 0.5 < best_threshold([0.3, 0.4, 0.7, 0.8], [False, False, True, True]) < 0.7


def test_golden_set_integrity():
    items = load_golden("all")
    assert len({i["id"] for i in items}) == len(items)
    for i in items:
        assert i["split"] in {"dev", "test"}
        assert i["expected_route"] in {"rag", "direct", "refuse"}
        assert set(i["relevant_sources"]) <= DOCS, i["id"]
        if i["category"] in {"company", "borderline"} and i["expected_route"] == "rag":
            assert i["relevant_sources"] and i["reference_answer"], i["id"]
        if i["category"] == "unanswerable":
            assert i["relevant_sources"] == [] and i["expected_route"] == "rag"
    assert {i["category"] for i in load_golden("test")} == {"company", "general", "adversarial",
                                                            "unanswerable", "borderline"}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def test_few_shot_examples_do_not_leak_into_eval_set():
    examples = [q for q, _, _ in ROUTER_EXAMPLES]
    examples += re.findall(r"Question: (.+)", critic_prompt.CRITIC_SYSTEM)
    for item in load_golden("all"):
        q = _tokens(item["question"])
        for ex in examples:
            e = _tokens(ex)
            assert len(q & e) / len(q | e) < 0.6, (item["id"], ex)
