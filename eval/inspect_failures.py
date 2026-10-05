"""List the items an eval run got wrong: misroutes, judge-incorrect or unfaithful answers, errors.

    python -m eval.inspect_failures                                   # eval/results/eval_test.json
    python -m eval.inspect_failures --file eval/results/eval_test_no-reflection.json
"""

import argparse
import json
from pathlib import Path

from eval.common import RESULTS_DIR


def failures(records: list[dict]) -> list[tuple[str, dict]]:
    out = []
    for r in records:
        reasons = []
        if r.get("error"):
            reasons.append("error")
        if r.get("route_decision") and r["route_decision"] != r["expected_route"]:
            reasons.append(f"misrouted ({r['expected_route']} -> {r['route_decision']})")
        judge = r.get("judge_final") or {}
        if judge and not judge.get("correct"):
            reasons.append("incorrect")
        if judge and not judge.get("faithful"):
            reasons.append("unfaithful")
        if reasons:
            out.append((", ".join(reasons), r))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, default=RESULTS_DIR / "eval_test.json")
    args = p.parse_args()
    if not args.file.exists():
        raise SystemExit(f"{args.file} not found. Run the eval first (make eval / tasks.ps1 eval).")
    records = json.loads(args.file.read_text(encoding="utf-8"))["records"]
    found = failures(records)
    for why, r in found:
        print(f"\n[{r['id']}] {r['category']}: {why}")
        print(f"  question: {r['question']}")
        print(f"  answer:   {r.get('answer') or r.get('error')}")
        if r.get("judge_final"):
            print(f"  judge:    {r['judge_final'].get('reason')}")
        if r.get("failed_criteria"):
            print(f"  critic failed: {', '.join(r['failed_criteria'])} (decision={r.get('decision')})")
    print(f"\n{len(found)} of {len(records)} items need a look")


if __name__ == "__main__":
    main()
