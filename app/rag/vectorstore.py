"""FAISS inner-product index plus a JSON sidecar for chunk text and build metadata.

The manifest records how the index was built (embedding model, query instruction,
chunking, corpus hashes). The retriever reads the embedding model from the manifest,
so query-time and build-time embeddings cannot silently diverge.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from app.rag.chunking import Chunk

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


class FAISSVectorStore:
    def __init__(self, dim: int):
        # Inner product on L2-normalised vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []
        self.manifest: dict = {}

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        embeddings = np.asarray(embeddings, dtype="float32")
        if len(embeddings) != len(chunks):
            raise ValueError(f"{len(embeddings)} embeddings for {len(chunks)} chunks")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        k = min(k, len(self))
        if k <= 0:
            return []
        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query, k)
        # FAISS pads missing results with -1; never let that index into self.chunks.
        return [(self.chunks[i], float(s)) for i, s in zip(indices[0], scores[0], strict=True) if i >= 0]

    def save(self, path: str | Path, manifest: dict) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / INDEX_FILE))
        with open(path / CHUNKS_FILE, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        self.manifest = {**manifest, "n_chunks": len(self.chunks), "dim": self.index.d}
        (path / MANIFEST_FILE).write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FAISSVectorStore":
        path = Path(path)
        manifest_path = path / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No index at {path}. Build it with: python -m app.rag.build_index")
        manifest = json.loads(manifest_path.read_text())
        index = faiss.read_index(str(path / INDEX_FILE))
        with open(path / CHUNKS_FILE, encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        if index.ntotal != len(chunks) or manifest.get("n_chunks") != len(chunks):
            raise ValueError(f"Corrupt index at {path}: {index.ntotal} vectors, "
                             f"{len(chunks)} chunks, manifest says {manifest.get('n_chunks')}")
        store = cls(index.d)
        store.index, store.chunks, store.manifest = index, chunks, manifest
        return store
