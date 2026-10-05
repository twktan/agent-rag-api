"""End-to-end agent evaluation on the golden set. Uses the OpenAI API, so it costs money:
it prints an estimate first and asks for confirmation (skip with --yes).

    python -m eval.run_eval --split test
    python -m eval.run_eval --split test --max-refinements 0 --tag no-reflection   # ablation

Measures routing (accuracy, per-class P/R/F1, confusion matrix), retrieval (Hit@k, MRR),
answer quality (judge-scored correctness + faithfulness, first draft vs final), abstention on
unanswerable questions, guardrails (attack block rate, false refusals), the refinement loop,
latency percentiles (end-to-end and per stage), and cost per query.
"""

import argparse
import asyncio
import sys
import time
from collections import Counter

from app.agent.graph import RAGAgent
from app.agent.summary import summarize_run
from app.config import get_settings
from app.llm.client import LLMClient
from app.llm.pricing import cost_usd
from app.prompts import PROMPT_VERSION
from app.prompts.common import REFUSAL_MESSAGE, is_abstention
from app.rag.retriever import Retriever
from eval.common import RESULTS_DIR, git_sha, load_golden, log_to_mlflow, utc_now, write_json
from eval.judge import Judge
from eval.metrics import classification_report, latency_summary, proportion
from eval.retrieval_eval import evaluate_retrieval

ROUTES = ["rag", "direct", "refuse"]
# Rough per-query token budgets (input, output) for the cost estimate only.
_EST_AGENT = {"rag": (5200, 450), "direct": (1100, 250), "refuse": (900, 30)}
_EST_JUDGE = (1800, 60)


def estimate_cost(items: list[dict], agent_model: str, judge_model: str) -> float:
    total = 0.0
    for item in items:
        tin, tout = _EST_AGENT[item["expected_route"]]
        total += cost_usd(agent_model, tin, tout) or 0.0
        if item["reference_answer"]:
            total += 2 * (cost_usd(judge_model, *_EST_JUDGE) or 0.0)  # first draft + final
    return total


async def run_agent(agent: RAGAgent, items: list[dict], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def one(item: dict) -> dict:
        async with sem:
            start = time.perf_counter()
            try:
                state = await agent.run(item["question"], request_id=item["id"])
            except Exception as exc:  # recorded, not fatal: errors are a metric too
                return {"item": item, "error": repr(exc)}
            return {"item": item, "state": state,
                    "summary": summarize_run(state, (time.perf_counter() - start) * 1000)}

    return await asyncio.gather(*(one(i) for i in items))


async def judge_answers(judge: Judge, records: list[dict], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    calls: list[dict] = []

    async def one(rec: dict) -> None:
        item, state = rec["item"], rec.get("state")
        if not state or not item["reference_answer"] or state["route"] != "rag":
            return
        attempts = state.get("attempts", [])
        docs = attempts[-1]["docs"] if attempts else state.get("docs", [])
        async with sem:
            final, call = await judge.judge(item["question"], item["reference_answer"], docs, state["answer"])
            calls.append(call.to_dict())
            rec["judge_final"] = final.model_dump()
            if attempts and attempts[0]["answer"] != state["answer"]:
                first, call = await judge.judge(item["question"], item["reference_answer"],
                                                attempts[0]["docs"], attempts[0]["answer"])
                calls.append(call.to_dict())
                rec["judge_first"] = first.model_dump()
            else:
                rec["judge_first"] = rec["judge_final"]

    await asyncio.gather(*(one(r) for r in records))
    return calls


def compute_metrics(records: list[dict]) -> dict:
    ok = [r for r in records if "state" in r]
    m: dict = {"n": len(records), "errors": len(records) - len(ok)}

    # Routing (guardrail + router decision, before any no-context fallback)
    y_true = [r["item"]["expected_route"] for r in ok]
    y_pred = [r["summary"]["route_decision"] for r in ok]
    m["routing"] = classification_report(y_true, y_pred, ROUTES)
    m["routing_by_category"] = {
        cat: proportion([r["summary"]["route_decision"] == r["item"]["expected_route"]
                         for r in ok if r["item"]["category"] == cat])
        for cat in sorted({r["item"]["category"] for r in ok})}

    # Answer quality (judge) on answerable questions that took the RAG path
    judged = [r for r in ok if "judge_final" in r]
    m["answer_correctness"] = proportion([r["judge_final"]["correct"] for r in judged])
    m["answer_faithfulness"] = proportion([r["judge_final"]["faithful"] for r in judged])
    m["first_draft_correctness"] = proportion([r["judge_first"]["correct"] for r in judged])
    m["first_draft_faithfulness"] = proportion([r["judge_first"]["faithful"] for r in judged])
    m["abstention_on_answerable"] = proportion([is_abstention(r["state"]["answer"]) for r in judged])

    unanswerable = [r for r in ok if r["item"]["category"] == "unanswerable"]
    m["abstention_on_unanswerable"] = proportion([is_abstention(r["state"]["answer"]) for r in unanswerable])

    # Guardrails
    attacks = [r for r in ok if r["item"]["expected_route"] == "refuse"]
    benign = [r for r in ok if r["item"]["expected_route"] != "refuse"]
    m["attack_block_rate"] = proportion([r["summary"]["route_decision"] == "refuse"
                                         or r["state"]["answer"] == REFUSAL_MESSAGE for r in attacks])
    m["false_refusal_rate"] = proportion([r["summary"]["route_decision"] == "refuse" for r in benign])
    m["blocked_by"] = dict(Counter(r["summary"]["blocked_reason"] for r in attacks))

    # Refinement loop (RAG path only)
    rag = [r for r in ok if r["state"]["route"] == "rag" and r["state"].get("attempts")]
    m["reflection"] = {
        "n": len(rag),
        "first_attempt_critic_pass": proportion([r["state"]["attempts"][0]["passed"] for r in rag]),
        "final_critic_pass": proportion([bool(r["state"]["quality"]["passed"]) for r in rag]),
        "refinement_triggered": proportion([r["summary"]["refinements"] > 0 for r in rag]),
        "mean_refinements": round(sum(r["summary"]["refinements"] for r in rag) / len(rag), 3) if rag else None,
        "decisions": dict(Counter(r["summary"]["decision"] for r in rag)),
        "citation_validity": proportion([r["state"]["attempts"][-1]["checks"].get("citations", False)
                                         for r in rag if not r["state"]["attempts"][-1]["abstained"]]),
    }

    # Latency and cost
    m["latency_ms"] = {"all": latency_summary([r["summary"]["latency_ms"] for r in ok])}
    for route in ROUTES:
        m["latency_ms"][route] = latency_summary([r["summary"]["latency_ms"] for r in ok
                                                  if r["summary"]["route"] == route])
    stages = sorted({s for r in ok for s in r["summary"]["timings_ms"]})
    m["stage_latency_ms"] = {s: latency_summary([r["summary"]["timings_ms"][s] for r in ok
                                                 if s in r["summary"]["timings_ms"]]) for s in stages}
    m["cost_usd_per_query"] = {route: round(sum(r["summary"]["cost_usd"] for r in ok if r["summary"]["route"] == route)
                                            / max(1, sum(r["summary"]["route"] == route for r in ok)), 6)
                               for route in ROUTES}
    m["cost_usd_per_query"]["all"] = round(sum(r["summary"]["cost_usd"] for r in ok) / max(1, len(ok)), 6)
    m["llm_calls_per_query"] = {
        route: round(sum(r["summary"]["llm_calls"] for r in ok if r["summary"]["route"] == route)
                     / max(1, sum(r["summary"]["route"] == route for r in ok)), 2)
        for route in ROUTES}
    return m


def slim(rec: dict) -> dict:
    item, s = rec["item"], rec.get("summary", {})
    out = {"id": item["id"], "category": item["category"], "expected_route": item["expected_route"],
           "question": item["question"], "error": rec.get("error")}
    if "state" in rec:
        out |= {"route_decision": s["route_decision"], "route": s["route"], "answer": rec["state"]["answer"],
                "sources": s["sources"], "decision": s["decision"], "refinements": s["refinements"],
                "failed_criteria": s["failed_criteria"], "blocked_reason": s["blocked_reason"],
                "latency_ms": s["latency_ms"], "timings_ms": s["timings_ms"], "cost_usd": s["cost_usd"],
                "judge_final": rec.get("judge_final"), "judge_first": rec.get("judge_first")}
    return out


def main() -> None:
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", choices=["dev", "test", "all"])
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N items (smoke test)")
    p.add_argument("--concurrency", type=int, default=1, help=">1 is faster but inflates latency numbers")
    p.add_argument("--max-refinements", type=int, default=settings.max_refinements)
    p.add_argument("--tag", default="", help="suffix for the results file, e.g. no-reflection")
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    args = p.parse_args()

    if not settings.openai_api_key:
        sys.exit("OPENAI_API_KEY is not set (export it or put it in .env).")
    items = load_golden(args.split)[: args.limit]
    estimate = estimate_cost(items, settings.llm_model, settings.judge_model)
    print(f"{len(items)} items, agent={settings.llm_model}, judge={settings.judge_model}, "
          f"estimated cost ~${estimate:.2f}")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        sys.exit("Aborted.")

    settings.max_refinements = args.max_refinements
    retriever = Retriever.from_index_dir(settings.index_dir, settings.min_similarity, settings.embed_threads)
    agent = RAGAgent(LLMClient(settings), retriever, settings)

    async def evaluate() -> tuple[list[dict], list[dict]]:
        # One event loop for everything: async HTTP clients must not outlive their loop.
        recs = await run_agent(agent, items, args.concurrency)
        return recs, await judge_answers(Judge(settings), recs, concurrency=4)

    start = time.perf_counter()
    records, judge_calls = asyncio.run(evaluate())
    metrics = compute_metrics(records)
    metrics["retrieval"] = evaluate_retrieval(retriever, items)["summary"]
    metrics["judge_cost_usd"] = round(sum(c["cost_usd"] or 0 for c in judge_calls), 4)
    metrics["agent_cost_usd_total"] = round(sum(r["summary"]["cost_usd"] for r in records if "summary" in r), 4)

    meta = {"split": args.split, "n": len(items), "git_sha": git_sha(), "timestamp": utc_now(),
            "llm_model": settings.llm_model, "judge_model": settings.judge_model,
            "prompt_version": PROMPT_VERSION, "max_refinements": args.max_refinements,
            "top_k": settings.top_k, "min_similarity": settings.min_similarity,
            "index_hash": retriever.store.manifest.get("index_hash"),
            "wall_clock_s": round(time.perf_counter() - start, 1), "concurrency": args.concurrency}
    name = f"eval_{args.split}" + (f"_{args.tag}" if args.tag else "")
    out = RESULTS_DIR / f"{name}.json"
    write_json(out, {"meta": meta, "metrics": metrics, "records": [slim(r) for r in records]})

    r = metrics
    print(f"routing acc={r['routing']['accuracy']['value']:.3f}  macro-F1={r['routing']['macro_f1']:.3f}")
    print(f"hit@3={r['retrieval']['hit@3']['value']:.3f}  MRR={r['retrieval']['mrr']:.3f}")
    print(f"correctness={r['answer_correctness']['value']}  faithfulness={r['answer_faithfulness']['value']}"
          f"  (first draft: {r['first_draft_correctness']['value']} / {r['first_draft_faithfulness']['value']})")
    print(f"abstain on unanswerable={r['abstention_on_unanswerable']['value']}  "
          f"attack block={r['attack_block_rate']['value']}  false refusal={r['false_refusal_rate']['value']}")
    print(f"latency p50={r['latency_ms']['all']['p50']}ms p95={r['latency_ms']['all']['p95']}ms  "
          f"cost/query=${r['cost_usd_per_query']['all']:.5f}  errors={r['errors']}")
    print(f"Wrote {out}")
    log_to_mlflow(name, meta, {
        "routing_accuracy": r["routing"]["accuracy"]["value"], "routing_macro_f1": r["routing"]["macro_f1"],
        "hit_at_3": r["retrieval"]["hit@3"]["value"], "mrr": r["retrieval"]["mrr"],
        "answer_correctness": r["answer_correctness"]["value"],
        "answer_faithfulness": r["answer_faithfulness"]["value"],
        "abstention_on_unanswerable": r["abstention_on_unanswerable"]["value"],
        "attack_block_rate": r["attack_block_rate"]["value"], "false_refusal_rate": r["false_refusal_rate"]["value"],
        "latency_p50_ms": r["latency_ms"]["all"].get("p50"), "latency_p95_ms": r["latency_ms"]["all"].get("p95"),
        "cost_per_query_usd": r["cost_usd_per_query"]["all"]}, out)


if __name__ == "__main__":
    main()
