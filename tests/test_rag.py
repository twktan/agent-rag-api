"""Chunking and vector store tests (no model download: random unit vectors)."""

import numpy as np
import pytest

from app.rag.chunking import Chunk, Document, char_chunks, chunk_corpus, load_corpus, title_from_source, word_chunks
from app.rag.vectorstore import FAISSVectorStore


def test_word_chunks_cover_text_with_overlap_and_whole_words():
    words = [f"w{i}" for i in range(250)]
    chunks = word_chunks(" ".join(words), size=100, overlap=20)
    assert [len(c.split()) for c in chunks] == [100, 100, 90]
    assert chunks[1].split()[:20] == chunks[0].split()[-20:]
    assert chunks[-1].split()[-1] == "w249"


def test_char_chunks_match_v1_behaviour():
    assert char_chunks("abcdefghij", size=4, overlap=1) == ["abcd", "defg", "ghij", "j"]


@pytest.mark.parametrize("size,overlap", [(10, 10), (10, 12), (10, -1)])
def test_invalid_overlap_rejected(size, overlap):
    with pytest.raises(ValueError):
        word_chunks("a b c", size, overlap)


def test_chunk_header_and_title():
    assert title_from_source("hr_benefits_policies") == "HR Benefits Policies"
    chunks = chunk_corpus([Document("devex_internal_tools", "Forge builds things")], "word", 50, 10, True)
    assert chunks[0].text == "DevEx Internal Tools: Forge builds things"


def test_load_corpus_is_sorted(tmp_path):
    for name in ("b.txt", "a.txt", "c.md"):
        (tmp_path / name).write_text(f"text {name}")
    assert [d.source for d in load_corpus(tmp_path)] == ["a", "b"]


def _unit(rng, n, d=8):
    v = rng.normal(size=(n, d)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _store(n=5):
    vectors = _unit(np.random.default_rng(0), n)
    store = FAISSVectorStore(dim=8)
    store.add(vectors, [Chunk(source=f"doc{i}", chunk_id=0, text=f"t{i}") for i in range(n)])
    return store, vectors


def test_search_returns_sorted_scores_and_self_match():
    store, vectors = _store()
    hits = store.search(vectors[2], k=3)
    assert hits[0][0].source == "doc2" and hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert [s for _, s in hits] == sorted([s for _, s in hits], reverse=True)


def test_k_larger_than_index_does_not_return_padding():
    store, vectors = _store(n=2)
    hits = store.search(vectors[0], k=10)
    assert len(hits) == 2 and {c.source for c, _ in hits} == {"doc0", "doc1"}


def test_save_load_roundtrip_and_corruption_check(tmp_path):
    store, vectors = _store()
    store.save(tmp_path, {"embedding_model": "fake"})
    loaded = FAISSVectorStore.load(tmp_path)
    assert loaded.manifest["n_chunks"] == 5 and loaded.manifest["embedding_model"] == "fake"
    assert loaded.search(vectors[1], 1)[0][0].source == "doc1"
    (tmp_path / "chunks.jsonl").write_text("")
    with pytest.raises(ValueError, match="Corrupt"):
        FAISSVectorStore.load(tmp_path)


def test_missing_index_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        FAISSVectorStore.load(tmp_path / "nope")
