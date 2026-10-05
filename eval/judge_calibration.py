"""Check the LLM judge against your own labels (agreement + Cohen's kappa).

    python -m eval.judge_calibration export     # writes eval/results/judge_calibration.csv
    # open the CSV, fill human_correct with TRUE/FALSE, save it (as "CSV UTF-8" in Excel)
    python -m eval.judge_calibration score

The CSV deliberately omits the judge's verdict, so labelling is blind; `score` joins your
labels with the judge's verdicts from the eval results by item id.
"""

import argparse
import csv
import json
import random
from pathlib import Path

from eval.common import RESULTS_DIR, load_golden, write_json
from eval.metrics import cohens_kappa

CSV_PATH = RESULTS_DIR / "judge_calibration.csv"
FIELDS = ["id", "question", "reference_answer", "answer", "human_correct"]
_TRUE, _FALSE = {"true", "t", "yes", "y", "1"}, {"false", "f", "no", "n", "0"}


def _judged(results: Path) -> dict[str, dict]:
    records = json.loads(results.read_text(encoding="utf-8"))["records"]
    return {r["id"]: r for r in records if r.get("judge_final")}


def export(results: Path, n: int, seed: int) -> None:
    judged = _judged(results)
    references = {i["id"]: i["reference_answer"] for i in load_golden("all")}
    ids = sorted(judged)
    random.Random(seed).shuffle(ids)
    # utf-8-sig so Excel on Windows detects the encoding instead of garbling non-ASCII text.
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for item_id in ids[:n]:
            writer.writerow({"id": item_id, "question": judged[item_id]["question"],
                             "reference_answer": references.get(item_id, ""),
                             "answer": judged[item_id]["answer"], "human_correct": ""})
    print(f"Wrote {min(n, len(ids))} rows to {CSV_PATH}. Fill in human_correct (TRUE/FALSE), "
          f"comparing each answer with the reference, then run: python -m eval.judge_calibration score")


def _parse(value: str) -> bool | None:
    v = value.strip().lower()
    return True if v in _TRUE else False if v in _FALSE else None


def score(results: Path) -> None:
    judged = _judged(results)
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:  # Excel may save with ";" as the delimiter in some locales
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.DictReader(f, dialect=dialect))
    pairs = [(_parse(r["human_correct"]), judged[r["id"]]["judge_final"]["correct"])
             for r in rows if r["id"] in judged]
    labelled = [(h, j) for h, j in pairs if h is not None]
    if not labelled:
        raise SystemExit(f"No labels found in {CSV_PATH}: fill the human_correct column first.")
    human, judge = [h for h, _ in labelled], [j for _, j in labelled]
    agreement = sum(h == j for h, j in labelled) / len(labelled)
    kappa = cohens_kappa(human, judge)
    print(f"n={len(labelled)}  agreement={agreement:.2f}  cohen_kappa={kappa:.2f}  "
          f"(skipped {len(pairs) - len(labelled)} unlabelled)")
    print(f"judge said correct: {sum(judge)}/{len(judge)}   you said correct: {sum(human)}/{len(human)}")
    disagreements = [r["id"] for r in rows if r["id"] in judged and _parse(r["human_correct"]) is not None
                     and _parse(r["human_correct"]) != judged[r["id"]]["judge_final"]["correct"]]
    if disagreements:
        print("disagreements (worth reading):", ", ".join(disagreements))
    write_json(RESULTS_DIR / "judge_calibration.json",
               {"n": len(labelled), "agreement": round(agreement, 4), "cohen_kappa": round(kappa, 4),
                "disagreements": disagreements, "results_file": results.name})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["export", "score"])
    p.add_argument("--results", type=Path, default=RESULTS_DIR / "eval_test.json")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not args.results.exists():
        raise SystemExit(f"{args.results} not found. Run the eval first.")
    if args.action == "export":
        export(args.results, args.n, args.seed)
    else:
        score(args.results)


if __name__ == "__main__":
    main()
