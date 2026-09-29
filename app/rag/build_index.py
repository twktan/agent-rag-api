"""Build the FAISS index from data/docs.

    python -m app.rag.build_index                 # uses Settings / env vars
    python -m app.rag.build_index --chunker char --chunk-size 500 --chunk-overlap 100
"""

import argparse
import hashlib
import logging
import time
from datetime import UTC, datetime

from app.config import get_settings
from app.rag.chunking import chunk_corpus, load_corpus
from app.rag.embed import Embedder
from app.rag.vectorstore import FAISSVectorStore

logger = logging.getLogger(__name__)


def build_store(docs_dir: str, embedder: Embedder, chunker: str, chunk_size: int,
                chunk_overlap: int, chunk_header: bool) -> FAISSVectorStore:
    docs = load_corpus(docs_dir)
    if not docs:
        raise FileNotFoundError(f"No .txt documents found in {docs_dir}")
    chunks = chunk_corpus(docs, chunker, chunk_size, chunk_overlap, chunk_header)
    store = FAISSVectorStore(embedder.dim)
    store.add(embedder.embed_documents([c.text for c in chunks]), chunks)
    store.manifest = {
        "embedding_model": embedder.model_name,
        "query_instruction": embedder.query_instruction,
        "chunker": chunker,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_header": chunk_header,
        "n_docs": len(docs),
        "doc_sha256": {d.source: hashlib.sha256(d.text.encode()).hexdigest()[:16] for d in docs},
    }
    corpus = "".join(store.manifest["doc_sha256"].values()) + repr(sorted(store.manifest.items()))
    store.manifest["index_hash"] = hashlib.sha256(corpus.encode()).hexdigest()[:12]
    return store


def main() -> None:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", default=s.docs_dir)
    p.add_argument("--out", default=s.index_dir)
    p.add_argument("--model", default=s.embedding_model)
    p.add_argument("--query-instruction", default=s.query_instruction)
    p.add_argument("--chunker", choices=["word", "char"], default=s.chunker)
    p.add_argument("--chunk-size", type=int, default=s.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=s.chunk_overlap)
    p.add_argument("--header", action=argparse.BooleanOptionalAction, default=s.chunk_header,
                   help="prefix each chunk with its document title")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start = time.perf_counter()
    embedder = Embedder(args.model, args.query_instruction, num_threads=s.embed_threads)
    store = build_store(args.docs, embedder, args.chunker, args.chunk_size, args.chunk_overlap,
                        chunk_header=args.header)
    store.save(args.out, {**store.manifest, "built_at": datetime.now(UTC).isoformat()})
    logger.info("Indexed %d chunks from %d docs into %s in %.1fs (index_hash=%s)",
                len(store), store.manifest["n_docs"], args.out, time.perf_counter() - start,
                store.manifest["index_hash"])


if __name__ == "__main__":
    main()
