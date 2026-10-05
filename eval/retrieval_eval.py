"""Offline retrieval evaluation — no LLM calls, no API key, runs in CI.

    python -m eval.retrieval_eval --split test
    python -m eval.retrieval_eval --split test --min-hit-at-3 0.85   # CI quality gate

Hit@k: share of answerable questions where a chunk from a relevant source document
appears in the top-k chunks (exactly what the generator sees when top_k=k).
Also measures query-side retrieval latency and fits a similarity threshold on the dev
split (used for `MIN_SIMILARITY` and as a free, non-LLM routing baseline).
"""

import argparse
import sys
import time

from app.config import get_settings
from app.rag.retriever import Retriever
from eval.common import RESULTS_DIR, git_sha, load_golden, log_to_mlflow, utc_now, write_json
from eval.metrics import (
    best_threshold,
    first_relevant_rank,
    latency_summary,
    mean_reciprocal_rank,
    proportion,
)

KS = (1, 3, 5)
MAX_K = 10


def evaluate_retrieval(retriever: Retriever, items: list[dict]) -> dict:
    ranks, latencies, per_item = [], [], []
    for item in items:
        if not item["relevant_sources"] or item["expected_route"] != "rag":
            continue
        start = time.perf_counter()
        hits = retriever.search(item["question"], MAX_K)
        latencies.append((time.perf_counter() - start) * 1000)
        rank = first_relevant_rank([h.source for h in hits], item["relevant_sources"])
        ranks.append(rank)
        per_item.append({"id": item["id"], "rank": rank, "top1_source": hits[0].source,
                         "top1_score": hits[0].score})
    summary = {f"hit@{k}": proportion([r is not None and r <= k for r in ranks]) for k in KS}
    summary["mrr"] = round(mean_reciprocal_rank(ranks), 4)
    summary["latency_ms"] = latency_summary(latencies)
    return {"summary": summary, "per_item": per_item}


def top1_scores(retriever: Retriever, items: list[dict]) -> tuple[list[float], list[bool]]:
    """Top-1 similarity for rag vs direct questions (adversarial items excluded)."""
    scores, is_rag = [], []
    for item in items:
        if item["expected_route"] in ("rag", "direct"):
            scores.append(retriever.search(item["question"], 1)[0].score)
            is_rag.append(item["expected_route"] == "rag")
    return scores, is_rag


def similarity_router_baseline(retriever: Retriever) -> dict:
    """Fit `route = rag if top1 >= t` on dev, report on test: a free baseline for the LLM router."""
    dev_scores, dev_labels = top1_scores(retriever, load_golden("dev"))
    test_scores, test_labels = top1_scores(retriever, load_golden("test"))
    t = best_threshold(dev_scores, dev_labels)
    rag_scores = [s for s, y in zip(dev_scores, dev_labels, strict=True) if y]
    direct_scores = [s for s, y in zip(dev_scores, dev_labels, strict=True) if not y]
    return {
        "threshold_fit_on_dev": round(t, 4),
        "test_accuracy": proportion([(s >= t) == y for s, y in zip(test_scores, test_labels, strict=True)]),
        "dev_top1_rag": {"min": min(rag_scores), "mean": round(sum(rag_scores) / len(rag_scores), 4)},
        "dev_top1_direct": {"max": max(direct_scores),
                            "mean": round(sum(direct_scores) / len(direct_scores), 4)},
    }


def main() -> None:
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", choices=["dev", "test", "all"])
    p.add_argument("--index-dir", default=settings.index_dir)
    p.add_argument("--min-hit-at-3", type=float, default=None, help="fail (exit 1) below this")
    args = p.parse_args()

    retriever = Retriever.from_index_dir(args.index_dir, num_threads=settings.embed_threads)
    result = evaluate_retrieval(retriever, load_golden(args.split))
    result["similarity_router"] = similarity_router_baseline(retriever)
    result["meta"] = {"split": args.split, "git_sha": git_sha(), "timestamp": utc_now(),
                      "index": {k: v for k, v in retriever.store.manifest.items() if k != "doc_sha256"}}
    out = RESULTS_DIR / f"retrieval_{args.split}.json"
    write_json(out, result)

    s = result["summary"]
    print(f"Retrieval ({args.split}, n={s['hit@1']['n']}): " +
          "  ".join(f"{k}={s[k]['value']:.3f}" for k in (f"hit@{k}" for k in KS)) +
          f"  MRR={s['mrr']:.3f}  p50={s['latency_ms']['p50']}ms  p95={s['latency_ms']['p95']}ms")
    sr = result["similarity_router"]
    print(f"Similarity-threshold router: t={sr['threshold_fit_on_dev']} "
          f"test acc={sr['test_accuracy']['value']:.3f}")
    print(f"Wrote {out}")
    log_to_mlflow(f"retrieval-{args.split}", result["meta"]["index"],
                  {"hit_at_1": s["hit@1"]["value"], "hit_at_3": s["hit@3"]["value"],
                   "hit_at_5": s["hit@5"]["value"], "mrr": s["mrr"],
                   "latency_p95_ms": s["latency_ms"]["p95"]}, out)

    if args.min_hit_at_3 is not None and s["hit@3"]["value"] < args.min_hit_at_3:
        print(f"FAIL: hit@3 {s['hit@3']['value']:.3f} < gate {args.min_hit_at_3}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
