"""Print golden-set items for human review, and search the corpus for a term.

    python -m eval.review_golden                          # all test items
    python -m eval.review_golden --split dev --category unanswerable
    python -m eval.review_golden --id hr_benefits         # items whose id starts with this
    python -m eval.review_golden --grep "per diem"        # which docs mention a term (with context)
    python -m eval.review_golden --out review.md          # write to a UTF-8 file instead (open in an editor)

Cross-platform replacement for ad-hoc `grep` over data/docs, so the review works the same
in PowerShell, bash or zsh.
"""

import argparse
import re
from collections.abc import Callable
from pathlib import Path

from eval.common import load_golden

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


def show_items(split: str, category: str | None, id_prefix: str | None, emit: Callable = print) -> int:
    items = [i for i in load_golden(split)
             if (category is None or i["category"] == category)
             and (id_prefix is None or i["id"].startswith(id_prefix))]
    for item in items:
        emit(f"\n[{item['id']}]  split={item['split']}  category={item['category']}  "
             f"expected_route={item['expected_route']}")
        emit(f"  sources:   {', '.join(item['relevant_sources']) or '(none)'}")
        emit(f"  question:  {item['question']}")
        emit(f"  reference: {item['reference_answer']}")
        if item.get("notes"):
            emit(f"  notes:     {item['notes']}")
    emit(f"\n{len(items)} item(s)")
    return len(items)


def grep_docs(term: str, context: int = 80, emit: Callable = print) -> int:
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    hits = 0
    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        hits += 1
        emit(f"\n{path.stem}  ({len(matches)} match{'es' if len(matches) > 1 else ''})")
        for m in matches[:3]:
            start, end = max(0, m.start() - context), min(len(text), m.end() + context)
            emit(f"  ...{text[start:end]}...")
    emit(f"\n'{term}' found in {hits} of {len(list(DOCS_DIR.glob('*.txt')))} docs")
    return hits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", choices=["dev", "test", "all"])
    p.add_argument("--category", choices=["company", "unanswerable", "general", "borderline", "adversarial"])
    p.add_argument("--id", dest="id_prefix", help="only items whose id starts with this")
    p.add_argument("--grep", help="search data/docs for this term instead of listing items")
    p.add_argument("--out", type=Path, help="write the output to this UTF-8 file instead of the terminal")
    args = p.parse_args()

    def run(emit: Callable) -> None:
        if args.grep:
            grep_docs(args.grep, emit=emit)
        else:
            show_items(args.split, args.category, args.id_prefix, emit=emit)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            run(lambda line="": f.write(f"{line}\n"))
        print(f"Wrote {args.out}")
    else:
        run(print)


if __name__ == "__main__":
    main()
