import os
from typing import List, Dict

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# Sliding window chunking
def simple_chunk_text(text: str,
                      chunk_size=CHUNK_SIZE,
                      overlap=CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start_idx = 0
    window_size = chunk_size - overlap
    while start_idx < len(text):
        chunk = text[start_idx:start_idx + chunk_size]
        start_idx += window_size
        chunks.append(chunk)

    return chunks


# Wrapper to chunk text
def ingest_directory(data_dir: str) -> List[Dict]:
    all_chunks = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)
            filetext = load_text_file(filepath)
            chunks = simple_chunk_text(filetext,
                                       chunk_size=CHUNK_SIZE,
                                       overlap=CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'metadata': {
                        'source': filename,
                        'chunk_id': i
                    }
                })
        else:
            continue
    return all_chunks
