"""Query -> scored chunks, with a similarity floor so off-topic queries get no context."""

from dataclasses import asdict, dataclass
from pathlib import Path

from app.rag.embed import Embedder
from app.rag.vectorstore import FAISSVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    chunk_id: int
    text: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


class Retriever:
    def __init__(self, store: FAISSVectorStore, embedder: Embedder, min_similarity: float = 0.0):
        self.store = store
        self.embedder = embedder
        self.min_similarity = min_similarity

    @classmethod
    def from_index_dir(cls, index_dir: str | Path, min_similarity: float = 0.0,
                       num_threads: int | None = None) -> "Retriever":
        store = FAISSVectorStore.load(index_dir)
        manifest = store.manifest
        embedder = Embedder(manifest["embedding_model"], manifest.get("query_instruction", ""),
                            num_threads=num_threads)
        if embedder.dim != store.index.d:
            raise ValueError(f"Embedding dim {embedder.dim} != index dim {store.index.d}")
        return cls(store, embedder, min_similarity)

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Top-k chunks with no similarity floor (used by offline evaluation)."""
        hits = self.store.search(self.embedder.embed_query(query), k)
        return [RetrievedChunk(c.source, c.chunk_id, c.text, round(s, 4)) for c, s in hits]

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return [c for c in self.search(query, k) if c.score >= self.min_similarity]
