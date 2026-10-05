"""Retrieval ablation: v1 baseline -> each change applied one at a time.

    python -m eval.ablate_retrieval

Configs are compared on the dev split (used for selection) and the held-out test split
(reported). Indexes are built in memory; nothing on disk is modified.
"""

from app.config import get_settings
from app.rag.build_index import build_store
from app.rag.embed import Embedder
from app.rag.retriever import Retriever
from eval.common import RESULTS_DIR, git_sha, load_golden, utc_now, write_json
from eval.retrieval_eval import evaluate_retrieval

INSTRUCTION = "Represent this sentence for searching relevant passages: "
V1, V15 = "BAAI/bge-small-en", "BAAI/bge-small-en-v1.5"

CONFIGS = [
    # name, model, query instruction, chunker, size, overlap, title header
    ("v1 baseline (char 500/100)", V1, "", "char", 500, 100, False),
    ("+ BGE query instruction", V1, INSTRUCTION, "char", 500, 100, False),
    ("+ bge-small-en-v1.5", V15, INSTRUCTION, "char", 500, 100, False),
    ("+ doc-title chunk header", V15, INSTRUCTION, "char", 500, 100, True),
    ("  variant: word chunks 120/30", V15, INSTRUCTION, "word", 120, 30, False),
    ("  variant: word 120/30 + header", V15, INSTRUCTION, "word", 120, 30, True),
    ("  variant: word 80/20 + header", V15, INSTRUCTION, "word", 80, 20, True),
    ("  variant: word 200/50 + header", V15, INSTRUCTION, "word", 200, 50, True),
]


def main() -> None:
    dev, test = load_golden("dev"), load_golden("test")
    threads = get_settings().embed_threads
    embedders: dict[str, Embedder] = {}
    rows = []
    for name, model, instruction, chunker, size, overlap, header in CONFIGS:
        if model not in embedders:
            embedders[model] = Embedder(model, num_threads=threads)
        embedder = embedders[model]
        embedder.query_instruction = instruction
        store = build_store("data/docs", embedder, chunker, size, overlap, header)
        retriever = Retriever(store, embedder)
        d = evaluate_retrieval(retriever, dev)["summary"]
        t = evaluate_retrieval(retriever, test)["summary"]
        rows.append({"config": name.strip(), "model": model, "query_instruction": bool(instruction),
                     "chunker": f"{chunker} {size}/{overlap}", "header": header, "n_chunks": len(store),
                     "dev": {"hit@3": d["hit@3"]["value"], "mrr": d["mrr"]},
                     "test": {k: t[k]["value"] for k in ("hit@1", "hit@3", "hit@5")} | {"mrr": t["mrr"]},
                     "test_n": t["hit@1"]["n"]})
        print(f"{name:34s} chunks={len(store):3d}  dev hit@3={d['hit@3']['value']:.3f} "
              f"mrr={d['mrr']:.3f} | test hit@1={t['hit@1']['value']:.3f} "
              f"hit@3={t['hit@3']['value']:.3f} hit@5={t['hit@5']['value']:.3f} mrr={t['mrr']:.3f}")

    best = max(rows, key=lambda r: (r["dev"]["mrr"], r["dev"]["hit@3"]))
    print(f"\nSelected on dev MRR: {best['config']}")
    write_json(RESULTS_DIR / "retrieval_ablation.json",
               {"rows": rows, "selected": best["config"], "git_sha": git_sha(), "timestamp": utc_now()})


if __name__ == "__main__":
    main()
