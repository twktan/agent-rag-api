"""Pure metric functions (no I/O), unit-tested in tests/test_eval_metrics.py."""

import math
from collections.abc import Iterable, Sequence

import numpy as np


def first_relevant_rank(retrieved_sources: Sequence[str], relevant: Iterable[str]) -> int | None:
    """1-based rank of the first retrieved chunk whose source is relevant, or None."""
    relevant = set(relevant)
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in relevant:
            return rank
    return None


def hit_at_k(ranks: Sequence[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks) if ranks else 0.0


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    return sum(1.0 / r for r in ranks if r) / len(ranks) if ranks else 0.0


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval; well-behaved for small n and proportions near 0 or 1."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def proportion(flags: Sequence[bool]) -> dict:
    n, k = len(flags), sum(bool(f) for f in flags)
    lo, hi = wilson_interval(k, n)
    return {"value": k / n if n else None, "ci95": [round(lo, 4), round(hi, 4)], "n": n}


def latency_summary(values_ms: Sequence[float]) -> dict:
    if not values_ms:
        return {"n": 0}
    arr = np.asarray(values_ms, dtype=float)
    return {"n": int(arr.size), "mean": round(float(arr.mean()), 1),
            "p50": round(float(np.percentile(arr, 50)), 1),
            "p95": round(float(np.percentile(arr, 95)), 1),
            "p99": round(float(np.percentile(arr, 99)), 1)}


def classification_report(y_true: Sequence[str], y_pred: Sequence[str],
                          labels: Sequence[str]) -> dict:
    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred, strict=True):
        confusion[t][p] += 1
    per_class = {}
    for c in labels:
        tp = confusion[c][c]
        fp = sum(confusion[t][c] for t in labels if t != c)
        fn = sum(confusion[c][p] for p in labels if p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": round(precision, 4), "recall": round(recall, 4),
                        "f1": round(f1, 4), "support": tp + fn}
    correct = sum(t == p for t, p in zip(y_true, y_pred, strict=True))
    return {"accuracy": proportion([t == p for t, p in zip(y_true, y_pred, strict=True)]),
            "macro_f1": round(sum(v["f1"] for v in per_class.values()) / len(labels), 4),
            "per_class": per_class, "confusion": confusion, "n_correct": correct}


def best_threshold(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Threshold t maximising accuracy of the rule (score >= t) == label."""
    candidates = sorted(set(scores))
    mids = [(a + b) / 2 for a, b in zip(candidates, candidates[1:], strict=False)] or candidates
    def acc(t: float) -> float:
        return sum((s >= t) == y for s, y in zip(scores, labels, strict=True)) / len(scores)
    return max(mids, key=acc)
