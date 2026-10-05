"""Print golden-set items for human review, and search the corpus for a term.

    python -m eval.review_golden                          # all test items
    python -m eval.review_golden --split dev --category unanswerable
    python -m eval.review_golden --id hr_benefits         # items whose id starts with this
    python -m eval.review_golden --grep "per diem"        # which docs mention a term (with context)

Cross-platform replacement for ad-hoc `grep` over data/docs, so the review works the same
in PowerShell, bash or zsh.
"""

import argparse
import re
from pathlib import Path

from eval.common import load_golden

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


def show_items(split: str, category: str | None, id_prefix: str | None) -> int:
    items = [i for i in load_golden(split)
             if (category is None or i["category"] == category)
             and (id_prefix is None or i["id"].startswith(id_prefix))]
    for item in items:
        print(f"\n[{item['id']}]  split={item['split']}  category={item['category']}  "
              f"expected_route={item['expected_route']}")
        print(f"  sources:   {', '.join(item['relevant_sources']) or '(none)'}")
        print(f"  question:  {item['question']}")
        print(f"  reference: {item['reference_answer']}")
        if item.get("notes"):
            print(f"  notes:     {item['notes']}")
    print(f"\n{len(items)} item(s)")
    return len(items)


def grep_docs(term: str, context: int = 80) -> int:
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    hits = 0
    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        hits += 1
        print(f"\n{path.stem}  ({len(matches)} match{'es' if len(matches) > 1 else ''})")
        for m in matches[:3]:
            start, end = max(0, m.start() - context), min(len(text), m.end() + context)
            print(f"  ...{text[start:end]}...")
    print(f"\n'{term}' found in {hits} of {len(list(DOCS_DIR.glob('*.txt')))} docs")
    return hits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", choices=["dev", "test", "all"])
    p.add_argument("--category", choices=["company", "unanswerable", "general", "borderline", "adversarial"])
    p.add_argument("--id", dest="id_prefix", help="only items whose id starts with this")
    p.add_argument("--grep", help="search data/docs for this term instead of listing items")
    args = p.parse_args()
    if args.grep:
        grep_docs(args.grep)
    else:
        show_items(args.split, args.category, args.id_prefix)


if __name__ == "__main__":
    main()
