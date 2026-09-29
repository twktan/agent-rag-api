"""Corpus loading and chunking.

The corpus files are single run-on paragraphs, so sentence/paragraph splitting has
nothing to work with. Word windows avoid cutting words in half, and an optional
document-title header gives every chunk the context its source file would otherwise
only carry in the filename.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

_SPECIAL_CASE = {"ai": "AI", "hr": "HR", "ml": "ML", "esg": "ESG", "devex": "DevEx", "tti": "TTI"}


@dataclass(frozen=True)
class Document:
    source: str  # filename stem, used as the citation id
    text: str


@dataclass(frozen=True)
class Chunk:
    source: str
    chunk_id: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_corpus(docs_dir: str | Path) -> list[Document]:
    # Sorted so that chunk order, and therefore the index, is reproducible across machines.
    paths = sorted(Path(docs_dir).glob("*.txt"))
    return [Document(source=p.stem, text=p.read_text(encoding="utf-8").strip()) for p in paths]


def title_from_source(source: str) -> str:
    words = source.replace("-", "_").split("_")
    return " ".join(_SPECIAL_CASE.get(w, w.capitalize()) for w in words)


def word_chunks(text: str, size: int, overlap: int) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError(f"need 0 <= overlap < size, got size={size} overlap={overlap}")
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def char_chunks(text: str, size: int, overlap: int) -> list[str]:
    """v1 sliding character window, kept verbatim so the baseline stays reproducible."""
    if not 0 <= overlap < size:
        raise ValueError(f"need 0 <= overlap < size, got size={size} overlap={overlap}")
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def chunk_corpus(docs: list[Document], chunker: str, size: int, overlap: int,
                 header: bool) -> list[Chunk]:
    split = {"word": word_chunks, "char": char_chunks}[chunker]
    chunks = []
    for doc in docs:
        for i, piece in enumerate(split(doc.text, size, overlap)):
            text = f"{title_from_source(doc.source)}: {piece}" if header else piece
            chunks.append(Chunk(source=doc.source, chunk_id=i, text=text))
    return chunks
